import torch
import torch.nn as nn


class StyleEncoder(nn.Module):
    def __init__(self, style_dim=3):
        super(StyleEncoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, style_dim),
            nn.ReLU(inplace=True)  # make style non-negative
        )

    def forward(self, low, high):
        """
        low:  (B,3,H,W)
        high: (B,3,H,W)
        """
        x = torch.cat([low, high], dim=1)  # (B,6,H,W)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        s = self.fc(x)
        return s
