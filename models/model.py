import torch
import torchvision
from torch import nn
from torchinfo import summary

from models.capsule_layer import CapsuleLayer


def build_model(arch, num_classes, input_size, trainable_layers, pretrained,
                num_primary_units, primary_unit_size, debug=False):
    # Load backbone
    if arch == "mobilenetv2":
        backbone_model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        feature_extractor = nn.Sequential(*list(backbone_model.features.children()))
    elif arch == "efficientnet_b0":
        backbone_model = torchvision.models.efficientnet_b0(pretrained=pretrained)
        feature_extractor = backbone_model.features
    elif arch == "shufflenet_v2_x1_0":
        backbone_model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
        feature_extractor = backbone_model.features
    else:
        raise ValueError("Unsupported architecture")

    # Freeze layers
    if pretrained:
        for param in backbone_model.parameters():
            param.requires_grad = False
        for param in list(backbone_model.parameters())[-trainable_layers:]:
            param.requires_grad = True

    # Dummy pass to get CNN output shape
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, input_size, input_size)
        cnn_out = feature_extractor(dummy_input)
        _, conv_out_channels, H, W = cnn_out.shape
        if debug:
            print(f"📦 CNN output shape: {cnn_out.shape}")

    # PrimaryCaps layer
    num_primary_units = num_primary_units  # 32
    primary_unit_size = primary_unit_size  # 8
    primary_caps = CapsuleLayer(
        in_units=0,
        in_channels=conv_out_channels,
        num_units=num_primary_units,
        unit_size=primary_unit_size,
        use_routing=False
    )

    # Get PrimaryCaps output shape
    with torch.no_grad():
        primary_out = primary_caps(cnn_out)
        _, in_channels_digit, in_units_digit = primary_out.shape
        if debug:
            print(f"🎯 PrimaryCaps output shape: {primary_out.shape}")

    # DigitCaps layer
    digit_caps = CapsuleLayer(
        in_units=in_units_digit,
        in_channels=in_channels_digit,
        num_units=num_classes,
        unit_size=primary_unit_size*2,
        use_routing=True
    )
    # Get PrimaryCaps output shape
    with torch.no_grad():
        digital_out = digit_caps(primary_out)
        if debug:
            print(f"🎯 DigitalCaps output shape: {digital_out.shape}")

    # Final model class
    class CapsuleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.primary_caps = primary_caps
            self.digit_caps = digit_caps

        def forward(self, x):
            x = self.feature_extractor(x)
            x = self.primary_caps(x)
            x = self.digit_caps(x)
            return x  # [B, num_classes, capsule_dim]

    model = CapsuleNet()

    # Print model summary
    summary(model, input_size=(1, 3, input_size, input_size))

    return model
