import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


# =====================================================================
# 1. ALGORITMA 1: SimpleCNN (Arsitektur dengan 2 kali Simple CNN - PM9)
# =====================================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_c=3, output=6):
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.fc = nn.Linear(in_features=50 * 5 * 5, out_features=output)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.fc(torch.flatten(x, 1))
        return x


# =====================================================================
# 2. ALGORITMA 2: ModifiedCNN (SimpleCNN yang diubah + Anti-Overfitting)
# =====================================================================
class ModifiedCNN(nn.Module):
    def __init__(self, input_c=3, output=6, dropout=0.5):
        super(ModifiedCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.convBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 2 * 2, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128, out_features=output)
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.classifier(x)
        return x


# =====================================================================
# 3. ALGORITMA 3: Deep Learning dari Luar (MobileNetV3-Small Pretrained)
# =====================================================================
class ExternalDL(nn.Module):
    def __init__(self, output=6, dropout=0.5):
        super(ExternalDL, self).__init__()
        base = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for param in base.features[:-2].parameters():
            param.requires_grad = False
        in_features = base.classifier[0].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, output),
        )
        self.model = base
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[-1] != 128:
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(x)
