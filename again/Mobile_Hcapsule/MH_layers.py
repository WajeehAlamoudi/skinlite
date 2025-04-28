import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MSharedEncoder(nn.Module):
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


class MCoarseFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        mobilenet = models.mobilenet_v2(pretrained=True)

        # Use MobileNet low-level block: 3 bottlenecks
        self.low_level = nn.Sequential(*list(mobilenet.features.children())[1:4])

        # Adapter: 64 → 32 to match MobileNet input
        self.adapter = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x):
        x = self.adapter(x)  # (B, 32, 224, 224)
        x = self.low_level(x)  # (B, 24, 112, 112)
        return x


class MMediumFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        mobilenet = models.mobilenet_v2(pretrained=True)

        # Fix: Include enough blocks to reach 96 channels
        self.middle_block = nn.Sequential(*list(mobilenet.features.children())[1:13])

        self.adapter = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x):
        x = self.adapter(x)  # (B, 32, 224, 224)
        x = self.middle_block(x)  # Expecting (B, 96, 28, 28)
        return x


class MFineFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        mobilenet = models.mobilenet_v2(pretrained=True)

        # Use full MobileNet features starting from block 1 to the end
        self.fine_block = nn.Sequential(*list(mobilenet.features.children())[1:])

        # Adapter to map SharedEncoder's output to expected MobileNet input
        self.adapter = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x_shared):
        x = self.adapter(x_shared)  # (B, 32, 224, 224)
        x = self.fine_block(x)  # Expecting (B, 1280, 14, 14)
        return x


class MCapsuleDecoder(nn.Module):
    def __init__(self, input_dim=16, output_shape=(3, 224, 224)):
        super().__init__()
        self.output_shape = output_shape
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_shape[0] * output_shape[1] * output_shape[2]),
            nn.Sigmoid()
        )

    def forward(self, digit_caps_output):
        # Select highest norm capsule (class-wise)
        v_norm = digit_caps_output.norm(dim=-1)  # [B, num_classes]
        class_idx = v_norm.argmax(dim=-1)  # [B]

        # Gather selected capsule vector for each sample
        B = digit_caps_output.size(0)
        selected = digit_caps_output[torch.arange(B), class_idx]  # [B, 16]

        out = self.decoder(selected)  # [B, C*H*W]
        return out.view(B, *self.output_shape)  # [B, C, H, W]


class MPrimaryCaps(nn.Module):
    def __init__(self, in_channels, capsule_dim=8, num_capsules=32, kernel_size=7, stride=2, padding=0):
        super(MPrimaryCaps, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

        # Shared convolutional layer producing all capsule maps
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def squash(self, x, dim=-1):
        # Non-linear squash function to ensure vector norm < 1
        norm = torch.norm(x, dim=dim, keepdim=True)
        scale = (norm ** 2) / (1.0 + norm ** 2)
        return scale * x / (norm + 1e-6)

    def forward(self, x):
        B = x.size(0)

        # Step 1: Convolution
        out = self.conv(x)  # [B, C×D, H, W]

        # Step 2: Reshape into [B, num_capsules, capsule_dim, H*W]
        out = out.view(B, self.num_capsules, self.capsule_dim, -1)

        # Step 3: Permute to [B, H*W, num_capsules, capsule_dim]
        out = out.permute(0, 3, 1, 2).contiguous()

        # Step 4: Merge all spatial capsules: [B, N, D]
        out = out.view(B, -1, self.capsule_dim)

        # Step 5: Squash activation
        return self.squash(out)


class MDigitCaps(nn.Module):
    def __init__(self, input_capsules, input_dim, num_classes, output_dim=16, routing_iters=2):
        super().__init__()
        self.input_capsules = input_capsules  # Number of primary capsules (N)
        self.input_dim = input_dim  # Dimension of each input capsule (e.g., 8)
        self.num_classes = num_classes  # Number of output capsules (classes)
        self.output_dim = output_dim  # Dimension of each output capsule (e.g., 16)
        self.routing_iters = routing_iters  # Number of routing iterations

        # Learnable transformation matrices: shape [1, N, num_classes, 16, 8]
        self.W = nn.Parameter(
            0.01 * torch.randn(1, input_capsules, num_classes, output_dim, input_dim)
        )

    def squash(self, x, dim=-1):
        norm = x.norm(dim=dim, keepdim=True)
        scale = (norm ** 2) / (1.0 + norm ** 2)
        return scale * x / (norm + 1e-6)

    def forward(self, x):
        B = x.size(0)  # batch size
        # Input shape: [B, N, 8]

        # Step 1: Unsqueeze input to [B, N, 1, 8, 1]
        x = x.unsqueeze(2).unsqueeze(-1)

        # Step 2: Expand W to match batch size: [B, N, num_classes, 16, 8]
        W = self.W.expand(B, -1, -1, -1, -1)

        # Step 3: Apply transformation: u_hat = W @ x → [B, N, num_classes, 16]
        u_hat = torch.matmul(W, x).squeeze(-1)

        # Step 4: Initialize logits (agreement scores): b_ij = 0
        b_ij = torch.zeros(B, self.input_capsules, self.num_classes, 1, device=x.device)

        # Step 5: Dynamic Routing
        for r in range(self.routing_iters):
            # Routing coefficients: [B, N, num_classes, 1]
            c_ij = F.softmax(b_ij, dim=2)

            # Weighted sum over input capsules: [B, 1, num_classes, 16]
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # Apply squash to get output capsule vectors: [B, 1, num_classes, 16]
            v_j = self.squash(s_j, dim=-1)

            # Update logits (except last iter): [B, N, num_classes, 1]
            if r < self.routing_iters - 1:
                b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(1)  # Final output: [B, num_classes, 16]


class MH1CapsCoarse(nn.Module):
    def __init__(self, num_classes=2):
        super(MH1CapsCoarse, self).__init__()

        self.feature_extractor = MCoarseFeatureExtractor()  # Output: [B, 24, 112, 112]

        self.primary_caps = MPrimaryCaps(
            in_channels=24,
            capsule_dim=8,
            num_capsules=32,
            kernel_size=7,
            stride=2
        )  # Output: [B, 89888, 8]

        self.digit_caps = MDigitCaps(
            input_capsules=89888,
            input_dim=8,
            num_classes=num_classes,
            output_dim=16,
            routing_iters=2
        )  # Output: [B, num_classes, 16]

        self.decoder = MCapsuleDecoder(input_dim=16, output_shape=(3, 224, 224))

    def forward(self, x_shared):
        x = self.feature_extractor(x_shared)  # [B, 24, 112, 112]
        x = self.primary_caps(x)  # [B, 89888, 8]
        digit_caps_out = self.digit_caps(x)  # [B, num_classes, 16]
        recon = self.decoder(digit_caps_out)
        return digit_caps_out, recon


class MH2CapsMedium(nn.Module):
    def __init__(self, num_classes=3):
        super(MH2CapsMedium, self).__init__()

        self.feature_extractor = MMediumFeatureExtractor()  # Output: [B, 96, 28, 28]

        self.primary_caps = MPrimaryCaps(
            in_channels=96,
            capsule_dim=8,
            num_capsules=32,
            kernel_size=7,
            stride=2
        )  # Output: [B, 3872, 8]

        self.digit_caps = MDigitCaps(
            input_capsules=3872,
            input_dim=8,
            num_classes=num_classes,
            output_dim=16,
            routing_iters=2
        )  # Output: [B, num_classes, 16]

        self.decoder = MCapsuleDecoder(input_dim=16, output_shape=(3, 224, 224))

    def forward(self, x_shared):
        x = self.feature_extractor(x_shared)  # [B, 96, 28, 28]
        x = self.primary_caps(x)  # [B, 3872, 8]
        digit_caps_out = self.digit_caps(x)  # [B, num_classes, 16]
        recon = self.decoder(digit_caps_out)
        return digit_caps_out, recon


class MH3CapsFine(nn.Module):
    def __init__(self, num_classes=7):
        super(MH3CapsFine, self).__init__()

        self.feature_extractor = MFineFeatureExtractor()  # Output: [B, 1280, 14, 14]

        self.primary_caps = MPrimaryCaps(
            in_channels=1280,
            capsule_dim=8,
            num_capsules=32,
            kernel_size=7,
            stride=2
        )  # Output: [B, 512, 8]

        self.digit_caps = MDigitCaps(
            input_capsules=512,
            input_dim=8,
            num_classes=num_classes,
            output_dim=16,
            routing_iters=2
        )  # Output: [B, num_classes, 16]

        self.decoder = MCapsuleDecoder(input_dim=16, output_shape=(3, 224, 224))

    def forward(self, x_shared):
        x = self.feature_extractor(x_shared)  # [B, 1280, 14, 14]
        x = self.primary_caps(x)  # [B, 512, 8]
        digit_caps_out = self.digit_caps(x)  # [B, num_classes, 16]
        recon = self.decoder(digit_caps_out)
        return digit_caps_out, recon


class MHCapsNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224)):
        super(MHCapsNet, self).__init__()

        self.shared_encoder = MSharedEncoder()

        self.h1 = MH1CapsCoarse(num_classes=2)
        self.h2 = MH2CapsMedium(num_classes=3)
        self.h3 = MH3CapsFine(num_classes=7)

        # Concatenated reconstructions: [B, 9, H, W] → Fuse to [B, 3, H, W]
        self.fuse_decoder = nn.Conv2d(9, 3, kernel_size=1)

    def forward(self, x):
        x_feat = self.shared_encoder(x)

        digit1, recon1 = self.h1(x_feat)  # recon1: [B, 3, 224, 224]
        digit2, recon2 = self.h2(x_feat)  # recon2: [B, 3, 224, 224]
        digit3, recon3 = self.h3(x_feat)  # recon3: [B, 3, 224, 224]

        recon_concat = torch.cat([recon1, recon2, recon3], dim=1)  # [B, 9, 224, 224]
        recon_final = self.fuse_decoder(recon_concat)  # [B, 3, 224, 224]

        return {
            "digit1": digit1,  # [B, 2, 16]
            "digit2": digit2,  # [B, 3, 16]
            "digit3": digit3,  # [B, 7, 16]
            "recon": recon_final  # [B, 3, 224, 224]
        }


def Mmargin_loss(vectors, labels, m_plus=0.9, m_minus=0.1, eta=0.5, class_weights=None):
    """
    Computes margin loss L_M^j for one hierarchy level (j).

    Args:
        vectors: Tensor [B, num_classes, dim] → capsule output vectors
        labels:  Tensor [B]                   → class indices
        :param class_weights:
    """
    # Compute vector norms: [B, num_classes]
    v_norm = torch.norm(vectors, dim=-1)

    # One-hot encoding of labels → [B, num_classes]
    T = F.one_hot(labels, num_classes=v_norm.size(1)).float()

    # Hinge loss terms
    positive_term = T * F.relu(m_plus - v_norm).pow(2)
    negative_term = eta * (1.0 - T) * F.relu(v_norm - m_minus).pow(2)

    L_M = positive_term + negative_term  # shape: [B, num_classes]
    # Apply weights if provided
    if class_weights is not None:
        weights = class_weights.to(vectors.device)[labels]  # [B]
        return (L_M.sum(dim=1) * weights).mean()
    else:
        return L_M.sum(dim=1).mean()


def Mdynamic_gamma(class_counts, accuracies, lambda_recon=0.0005):
    total_classes = sum(class_counts)
    rhos = [k / total_classes for k in class_counts]
    taus = [(1 - acc) * rho for acc, rho in zip(accuracies, rhos)]

    tau_sum = sum(taus)
    gamma = [(1 - lambda_recon) * tau / tau_sum for tau in taus]
    return gamma


def Mclassification_loss(digit1, digit2, digit3, label1, label2, label3,
                         gamma=[1 / 3, 1 / 3, 1 / 3], m_plus=0.9, m_minus=0.1, eta=0.5, class_weights=None):
    L1 = Mmargin_loss(digit1, label1, m_plus, m_minus, eta, class_weights=class_weights)
    L2 = Mmargin_loss(digit2, label2, m_plus, m_minus, eta, class_weights=class_weights)
    L3 = Mmargin_loss(digit3, label3, m_plus, m_minus, eta, class_weights=class_weights)
    LC = gamma[0] * L1 + gamma[1] * L2 + gamma[2] * L3
    return LC, (L1.item(), L2.item(), L3.item())


def Mtotal_hcapsnet_loss(outputs, labels, x_input, gamma, lambda_recon=0.0005, class_weights=None):
    digit1, digit2, digit3 = outputs["digit1"], outputs["digit2"], outputs["digit3"]
    label1, label2, label3 = labels
    recon = outputs["recon"]

    LC, (L1, L2, L3) = Mclassification_loss(
        digit1, digit2, digit3, label1, label2, label3, gamma=gamma, class_weights=class_weights
    )
    LR = F.mse_loss(recon, x_input)
    total = LC + lambda_recon * LR
    return total, LC.item(), LR.item(), (L1, L2, L3)
