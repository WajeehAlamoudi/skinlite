import torchvision
from torch import nn
import config


def build_model(arch, num_classes, trainable_layers, pretrained):

    if arch == "mobilenetv2":
        base = torchvision.models.mobilenet_v2(pretrained=pretrained).features
    elif arch == "efficientnet_b0":
        base = torchvision.models.efficientnet_b0(pretrained=pretrained).features
    elif arch == "shufflenet_v2_x1_0":
        base = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained).features
    else:
        raise ValueError("Unsupported architecture")

    # Freeze all layers, then unfreeze the last N
    for param in base.parameters():
        param.requires_grad = False
    for param in list(base.parameters())[-trainable_layers:]:
        param.requires_grad = True

    model = nn.Sequential(
        base,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(base[-1].out_channels, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model
