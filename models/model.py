import torch
import torch.nn as nn
import torchvision
from torchinfo import summary
import torch.nn.functional as F

import config
from models.capsule_layers import PrimaryCaps, DigitCaps, Decoder


def build_model(arch, num_classes, trainable_layers, pretrained,
                num_primary_units, primary_unit_size):
    # Load CNN backbone
    if arch == "mobilenetv2":
        backbone_model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(backbone_model.features.children()))
        backbone_out_channels = 1280
    elif arch == "efficientnet_b0":
        backbone_model = torchvision.models.efficientnet_b0(pretrained=pretrained)
        feature_extractor = backbone_model.features
        backbone_out_channels = 1280
    elif arch == "shufflenet_v2_x1_0":
        backbone_model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
        feature_extractor = nn.Sequential(
            backbone_model.conv1,
            backbone_model.maxpool,
            backbone_model.stage2,
            backbone_model.stage3,
            backbone_model.stage4,
            backbone_model.conv5
        )
        backbone_out_channels = 1024
    else:
        raise ValueError("Unsupported architecture")

    # Freeze layers if pretrained
    if pretrained:
        for param in backbone_model.parameters():
            param.requires_grad = False
        for param in list(backbone_model.parameters())[-trainable_layers:]:
            param.requires_grad = True

    # mobilenetv2 [B, 128, 7, 7]
    # Capsule layers
    primary_caps = PrimaryCaps(
        num_capsules=num_primary_units,
        in_channels=backbone_out_channels,
        out_channels=primary_unit_size
    )

    digit_caps = DigitCaps(
        num_caps=num_classes,
        previous_layer_nodes=primary_unit_size*7*7,
        in_channels=num_primary_units,
        out_channels=num_primary_units*2
    )

    decoder = Decoder(
        input_vector_length=num_primary_units*2,
        input_capsules=num_classes
    )

    # Full capsule net model
    class CapsuleNetModel(nn.Module):
        def __init__(self):
            super(CapsuleNetModel, self).__init__()
            self.feature_extractor = feature_extractor
            self.primary_caps = primary_caps
            self.digit_caps = digit_caps
            self.decoder = decoder
            # 🔸 Run a dummy forward pass to print all shapes once
            dummy_input = torch.randn(1, 3, config.run_config['IMAGE_SIZE'], config.run_config['IMAGE_SIZE'])
            with torch.no_grad():
                self._print_model_shapes(dummy_input)

        def _print_model_shapes(self, x):
            print(f"Input into {arch}:", x.shape)
            x = self.feature_extractor(x)
            print("After feature_extractor:", x.shape)

            x = self.primary_caps(x)
            print("After primary_caps:", x.shape)

            x = self.digit_caps(x)
            print("After digit_caps (pre-squeeze):", x.shape)

            if x.shape[2] == 1:
                x = x.squeeze(2)  # safely remove the 1-dim if it exists
                print("After digit_caps (squeezed):", x.shape)

            batch_size = x.size(0)
            num_classes = x.size(1)
            capsule_dim = x.size(2)

            dummy_y = torch.eye(num_classes, device=x.device)[[0]].expand(batch_size, num_classes)

            recon = self.decoder(x, dummy_y)
            print("Reconstruction:", recon.shape)

            class_preds = x.norm(dim=-1)
            print("Class predictions:", class_preds.shape)

        def forward(self, x, y=None):
            x = self.feature_extractor(x)
            x = self.primary_caps(x)
            x = self.digit_caps(x).squeeze(-1)  # [B, num_classes, 16]
            if y is None:
                # If not provided (e.g., during inference), use prediction
                class_preds = torch.norm(x, dim=-1)
                pred = class_preds.argmax(dim=1)
                y = F.one_hot(pred, num_classes=class_preds.size(1)).float()
            recon = self.decoder(x, y)
            return x, recon, y  # can also return logits = raw score

    summary(CapsuleNetModel())
    return CapsuleNetModel()
