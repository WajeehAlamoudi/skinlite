import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class CoarseFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.model(x)


class MediumFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        return x


class FineFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        return x


class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, capsule_dim=8, num_capsules=32, kernel_size=7, stride=2):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

        # One shared conv layer outputting (num_capsules * capsule_dim) channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=stride
        )

    def squash(self, x, dim=-1):
        norm = x.norm(dim=dim, keepdim=True)
        scale = (norm ** 2) / (1.0 + norm ** 2)
        return scale * x / (norm + 1e-6)

    def forward(self, x):
        B = x.size(0)
        out = self.conv(x)  # [B, num_capsules * capsule_dim, H, W]
        out = out.view(B, self.num_capsules, self.capsule_dim, -1)  # [B, C, D, HW]
        out = out.permute(0, 3, 1, 2).contiguous()  # [B, HW, C, D]
        out = out.view(B, -1, self.capsule_dim)  # [B, N, D]
        return self.squash(out)


class DigitCaps(nn.Module):
    def __init__(self, input_capsules, input_dim, num_classes, output_dim=16, routing_iters=2):
        super().__init__()
        self.input_capsules = input_capsules  # N
        self.input_dim = input_dim  # 8
        self.num_classes = num_classes  # e.g., 2 / 3 / 7
        self.output_dim = output_dim  # 16
        self.routing_iters = routing_iters

        # Transformation matrix W: [1, N, num_classes, output_dim, input_dim]
        self.W = nn.Parameter(
            0.01 * torch.randn(1, input_capsules, num_classes, output_dim, input_dim)
        )

    def squash(self, x, dim=-1):
        norm = x.norm(dim=dim, keepdim=True)
        scale = (norm ** 2) / (1.0 + norm ** 2)
        return scale * x / (norm + 1e-6)

    def forward(self, x):
        B = x.size(0)  # batch size

        # Step 1: [B, N, 1, 8, 1]
        x = x.unsqueeze(2).unsqueeze(-1)

        # Step 2: Expand W to [B, N, num_classes, 16, 8]
        W = self.W.expand(B, -1, -1, -1, -1)

        # Step 3: Compute u_hat: [B, N, num_classes, 16]
        u_hat = torch.matmul(W, x).squeeze(-1)

        # Step 4: Initialize routing logits b_ij = 0
        b_ij = torch.zeros(B, self.input_capsules, self.num_classes, 1, device=x.device)

        # Step 5: Dynamic Routing
        for r in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)  # softmax over num_classes
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # [B, 1, num_classes, 16]
            v_j = self.squash(s_j, dim=-1)  # [B, 1, num_classes, 16]
            if r < self.routing_iters - 1:
                b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(1)  # [B, num_classes, 16]


class Decoder(nn.Module):
    def __init__(self, num_classes, image_size=224, channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels

        self.decoder = nn.Sequential(
            nn.Linear(num_classes * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, channels * image_size * image_size),
            nn.Sigmoid()
        )

    def forward(self, digit_caps):
        # digit_caps: [B, num_classes, 16]
        B = digit_caps.size(0)
        x = digit_caps.view(B, -1)  # flatten to [B, num_classes * 16]
        x = self.decoder(x)
        return x.view(B, self.channels, self.image_size, self.image_size)


class H1CapsCoarse(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_extractor = CoarseFeatureExtractor()  # From previous step

        # After CoarseFeatureExtractor: [B, 128, 112, 112]
        self.primary_caps = PrimaryCaps(in_channels=128, capsule_dim=8, kernel_size=7, stride=2)

        # PrimaryCaps will output [B, N, 8], where N depends on input spatial size
        # We'll compute N dynamically on the first forward pass

        self.num_classes = num_classes
        self.routing_iters = 2
        self.input_capsules = None  # Set on first forward pass
        self.digit_caps = None  # Will be initialized lazily

        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, 128, 112, 112]
        primary = self.primary_caps(features)  # [B, N, 8]

        if self.digit_caps is None:
            N = primary.shape[1]  # Number of capsules
            self.input_capsules = N
            self.digit_caps = DigitCaps(
                input_capsules=N,
                input_dim=8,
                num_classes=self.num_classes,
                output_dim=16,
                routing_iters=self.routing_iters
            ).to(x.device)

        digit = self.digit_caps(primary)  # [B, 2, 16]
        recon = self.decoder(digit)  # [B, 3, 224, 224]
        return digit, recon


class H2CapsMedium(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.feature_extractor = MediumFeatureExtractor()  # 4-layer block

        self.primary_caps = PrimaryCaps(
            in_channels=256, capsule_dim=8, kernel_size=7, stride=2
        )

        self.num_classes = num_classes
        self.routing_iters = 2
        self.input_capsules = None  # to be set dynamically
        self.digit_caps = None

        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, 256, 56, 56]
        primary = self.primary_caps(features)  # [B, N, 8]

        if self.digit_caps is None:
            N = primary.size(1)
            self.input_capsules = N
            self.digit_caps = DigitCaps(
                input_capsules=N,
                input_dim=8,
                num_classes=self.num_classes,
                output_dim=16,
                routing_iters=self.routing_iters
            ).to(x.device)

        digit = self.digit_caps(primary)  # [B, 3, 16]
        recon = self.decoder(digit)  # [B, 3, 224, 224]
        return digit, recon


class H3CapsFine(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.feature_extractor = FineFeatureExtractor()  # 6-layer block

        self.primary_caps = PrimaryCaps(
            in_channels=512, capsule_dim=8, kernel_size=7, stride=2
        )

        self.num_classes = num_classes
        self.routing_iters = 2
        self.input_capsules = None
        self.digit_caps = None

        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, 512, 28, 28]
        primary = self.primary_caps(features)  # [B, N, 8]

        if self.digit_caps is None:
            N = primary.size(1)
            self.input_capsules = N
            self.digit_caps = DigitCaps(
                input_capsules=N,
                input_dim=8,
                num_classes=self.num_classes,
                output_dim=16,
                routing_iters=self.routing_iters
            ).to(x.device)

        digit = self.digit_caps(primary)  # [B, 7, 16]
        recon = self.decoder(digit)  # [B, 3, 224, 224]
        return digit, recon


class HCapsNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224)):
        super().__init__()
        self.shared_encoder = SharedEncoder()

        self.h1 = H1CapsCoarse(num_classes=2)
        self.h2 = H2CapsMedium(num_classes=3)
        self.h3 = H3CapsFine(num_classes=7)

        self.fuse_decoder = nn.Conv2d(9, 3, kernel_size=1)

    def forward(self, x):
        x_feat = self.shared_encoder(x)

        digit1, recon1 = self.h1(x_feat)
        digit2, recon2 = self.h2(x_feat)
        digit3, recon3 = self.h3(x_feat)

        recon_concat = torch.cat([recon1, recon2, recon3], dim=1)
        recon_final = self.fuse_decoder(recon_concat)

        return {
            "digit1": digit1,
            "digit2": digit2,
            "digit3": digit3,
            "recon": recon_final
        }


# ==== Loss fun ===


def margin_loss(vectors, labels, m_plus=0.9, m_minus=0.1, eta=0.5):
    """
    Computes margin loss L_M^j for one hierarchy level (j).

    Args:
        vectors: Tensor [B, num_classes, dim] → capsule output vectors
        labels:  Tensor [B]                   → class indices
    """
    # Compute vector norms: [B, num_classes]
    v_norm = torch.norm(vectors, dim=-1)

    # One-hot encoding of labels → [B, num_classes]
    T = F.one_hot(labels, num_classes=v_norm.size(1)).float()

    # Hinge loss terms
    positive_term = T * F.relu(m_plus - v_norm).pow(2)
    negative_term = eta * (1.0 - T) * F.relu(v_norm - m_minus).pow(2)

    L_M = positive_term + negative_term  # shape: [B, num_classes]
    return L_M.sum(dim=1).mean()  # mean over batch


def dynamic_gamma(class_counts, accuracies, lambda_recon=0.0005):
    total_classes = sum(class_counts)
    rhos = [k / total_classes for k in class_counts]
    taus = [(1 - acc) * rho for acc, rho in zip(accuracies, rhos)]

    tau_sum = sum(taus)
    gamma = [(1 - lambda_recon) * tau / tau_sum for tau in taus]
    return gamma


def classification_loss(digit1, digit2, digit3,label1, label2, label3,
                        gamma=[1/3, 1/3, 1/3],m_plus=0.9, m_minus=0.1, eta=0.5):
    L1 = margin_loss(digit1, label1, m_plus, m_minus, eta)
    L2 = margin_loss(digit2, label2, m_plus, m_minus, eta)
    L3 = margin_loss(digit3, label3, m_plus, m_minus, eta)
    LC = gamma[0]*L1 + gamma[1]*L2 + gamma[2]*L3
    return LC, (L1.item(), L2.item(), L3.item())


def total_hcapsnet_loss(outputs, labels, x_input, gamma, lambda_recon=0.0005):
    digit1, digit2, digit3 = outputs["digit1"], outputs["digit2"], outputs["digit3"]
    label1, label2, label3 = labels
    recon = outputs["recon"]

    LC, (L1, L2, L3) = classification_loss(
        digit1, digit2, digit3, label1, label2, label3, gamma=gamma
    )
    LR = F.mse_loss(recon, x_input)
    total = LC + lambda_recon * LR
    return total, LC.item(), LR.item(), (L1, L2, L3)
