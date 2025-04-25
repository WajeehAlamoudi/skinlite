import torch.nn as nn
from torchvision import models
import again.setting


class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetClassifier, self).__init__()
        base = models.mobilenet_v2(pretrained=True)
        self.features = base.features

        # Freeze all features
        for param in self.features.parameters():
            param.requires_grad = False

        # Unfreeze last 7 layers
        for layer in list(self.features.children())[-int(again.setting.TRAINABLE):]:
            for param in layer.parameters():
                param.requires_grad = True

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
