import torchvision
from torch import nn
from torchinfo import summary


def build_model(arch, num_classes, input_size, trainable_layers, pretrained):
    if arch == "mobilenetv2":
        base = torchvision.models.mobilenet_v2(pretrained=pretrained).features
    elif arch == "efficientnet_b0":
        base = torchvision.models.efficientnet_b0(pretrained=pretrained).features
    elif arch == "shufflenet_v2_x1_0":
        base = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained).features
    else:
        raise ValueError("Unsupported architecture")

    if pretrained:
        for param in base.parameters():
            param.requires_grad = False
        for param in list(base.parameters())[-trainable_layers:]:
            param.requires_grad = True
    else:
        # Train all layers from scratch
        for param in base.parameters():
            param.requires_grad = True

    model = nn.Sequential(
        base,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.6),
        nn.Linear(base[-1].out_channels, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    summary(model, input_size=(1, 3, input_size, input_size))

    return model


"""
# in_features = model.classifier[1].in_features
    model = nn.Sequential(
        base,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(base[-1].out_channels, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
"""
