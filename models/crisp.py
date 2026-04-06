import torch
import torch.nn as nn
from models.encoder import StyleEncoder
from models.style_decoder import StyleDecoder
from models.isp import ISP


class CRISP(nn.Module):
    def __init__(self, style_dim=5):
        super(CRISP, self).__init__()

        self.encoder = StyleEncoder(style_dim=style_dim)
        self.decoder = StyleDecoder(style_dim=style_dim)
        self.isp = ISP()

    def forward(self, low, high=None):
        """
        Training:
            low + high → learn style
        Testing:
            style vector provided manually (high=None)
        """

        if high is not None:
            # Training mode
            s = self.encoder(low, high)
        else:
            raise ValueError("During testing, provide style vector manually.")

        phi_residual = self.decoder(s)
        phi_residual = 0.5 * phi_residual  
        out = self.isp(low, phi_residual)

        return out, s
