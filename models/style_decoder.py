import torch
import torch.nn as nn


class StyleDecoder(nn.Module):
    def __init__(self, style_dim=5, hidden_dim=128, output_dim=19):
        super(StyleDecoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, s):
        return self.mlp(s)