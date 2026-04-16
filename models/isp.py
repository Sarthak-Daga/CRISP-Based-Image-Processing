'''
0   → digital gain (1)

1   → white balance R
2   → white balance B

3 –>11 → 3x3 CCM (9 values)

12 –>14 → CCM offset (3 values)

15  → gamma

16  → tone_s
17  → tone_p1
18  → tone_p2


'''

import torch
import torch.nn as nn


class ISP(nn.Module):
    def __init__(self, eps=1e-8):
        super(ISP, self).__init__()
        self.eps = eps

        # Define initial ISP parameters
        phi_init = torch.zeros(19)

        phi_init[0] = 1.2        # digital gain
        phi_init[1] = 1.0        # WB R
        phi_init[2] = 1.0        # WB B

        # CCM identity
        phi_init[3:12] = torch.tensor([
            1,0,0,
            0,1,0,
            0,0,1
        ])

        phi_init[12:15] = 0      # CCM offset

        phi_init[15] = 1/2.2     # gamma

        phi_init[16] = 3.0       # tone_s
        phi_init[17] = 2.0       # tone_p1
        phi_init[18] = 3.0       # tone_p2

        self.register_buffer("phi_init", phi_init)

    def forward(self, x, phi):
        phi = phi + self.phi_init
        phi0 = torch.clamp(phi[:,0:1], 0.3, 3.0)
        phi1 = torch.clamp(phi[:,1:2], 0.3, 3.0)
        phi2 = torch.clamp(phi[:,2:3], 0.3, 3.0)

        phi15 = torch.clamp(phi[:,15:16], 0.2, 2.5)
        
        phi_rest1 = phi[:,3:15]
        
        phi_rest2 = phi[:,16:]
        
        phi = torch.cat(
            [phi0, phi1, phi2, phi_rest1, phi15, phi_rest2],
            dim=1
        )

        """
        x: (B, 3, H, W)
        phi: (B, 19)
        """

        B, C, H, W = x.shape

        # -------------------------
        # 1. Digital Gain
        # -------------------------
        gain = phi[:, 0].view(B, 1, 1, 1)
        x = gain * x

        # -------------------------
        # 2. White Balance
        # -------------------------
        wb_r = phi[:, 1].view(B, 1, 1, 1)
        wb_b = phi[:, 2].view(B, 1, 1, 1)

        r = x[:, 0:1, :, :] * wb_r
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :] * wb_b

        x = torch.cat([r, g, b], dim=1)

        # -------------------------
        # 3. Color Correction Matrix (3x3)
        # -------------------------
        ccm = phi[:, 3:12].view(B, 3, 3)
        offset = phi[:, 12:15].view(B, 3, 1)

        x_flat = x.view(B, 3, -1)  # (B,3,H*W)

        x_ccm = torch.bmm(ccm, x_flat) + offset
        x = x_ccm.view(B, 3, H, W)

        # -------------------------
        # 4. Gamma Correction
        # -------------------------
        gamma = phi[:, 15].view(B, 1, 1, 1)
        x = torch.pow(torch.clamp(x, min=self.eps), gamma)

        # -------------------------
        # 5. Tone Mapping
        # -------------------------
        tone_s = phi[:, 16].view(B, 1, 1, 1)
        tone_p1 = phi[:, 17].view(B, 1, 1, 1)
        tone_p2 = phi[:, 18].view(B, 1, 1, 1)

        x1 = tone_s * torch.pow(torch.clamp(x, min=self.eps), tone_p1)
        x2 = (tone_s - 1) * torch.pow(torch.clamp(x, min=self.eps), tone_p2)

        x = x1 - x2

        return torch.clamp(x, 0.0, 1.0)
