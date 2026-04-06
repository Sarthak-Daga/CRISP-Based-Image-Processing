import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device).eval()

for p in vgg.parameters():
    p.requires_grad = False


def perceptual_loss(out, target):

    out = out.to(device)
    target = target.to(device)

    return nn.functional.l1_loss(vgg(out), vgg(target))

def color_loss(out, target):
    return torch.mean(
        torch.abs(
            out.mean(dim=[2, 3]) - target.mean(dim=[2, 3])
        )
    )

def contrast_loss(out, target):
    out_std = out.std(dim=[2,3])
    target_std = target.std(dim=[2,3])

    return torch.mean(torch.abs(out_std - target_std))

def ssim_loss(out, target):
    return 1 - ssim(out, target, data_range=1.0, size_average=True)

def total_loss(out, target):

    l1 = F.l1_loss(out, target)
    l2 = F.mse_loss(out, target)
    p = perceptual_loss(out, target)
    c = color_loss(out, target)
    k = contrast_loss(out, target)
    s = ssim_loss(out, target)

    return (0.5*l1 +0.3*l2 +0.2*s +0.1*p +0.1*c +0.05*k)