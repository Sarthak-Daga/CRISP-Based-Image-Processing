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
    return torch.mean(torch.abs(out - target))
    

def contrast_loss(out, target):
    return torch.mean(torch.abs(
        (out.max(dim=2)[0].max(dim=2)[0] - out.min(dim=2)[0].min(dim=2)[0]) -
        (target.max(dim=2)[0].max(dim=2)[0] - target.min(dim=2)[0].min(dim=2)[0])
    ))

def ssim_loss(out, target):
    return 1 - ssim(out, target, data_range=1.0, size_average=True)
def exposure_loss(out):
    mean = torch.mean(out, dim=[1,2,3])
    target = torch.full_like(mean, 0.5)
    return torch.mean((mean - target)**2)

def color_balance_loss(out, target):
    return torch.mean(torch.abs(torch.mean(out, dim=[2,3]) - torch.mean(target, dim=[2,3])))

def green_bias_loss(out, target):
    out_g = torch.mean(out[:,1,:,:])
    tgt_g = torch.mean(target[:,1,:,:])
    return torch.abs(out_g - tgt_g)

def vibrance_loss(out, target):
    return torch.mean(torch.abs(torch.std(out, dim=[2,3]) - torch.std(target, dim=[2,3])))


def total_loss(out, target):

    l1 = F.l1_loss(out, target)
    l2 = F.mse_loss(out, target)
    p = perceptual_loss(out, target)
    c = color_loss(out, target)
    k = contrast_loss(out, target)
    s = ssim_loss(out, target)
    e = exposure_loss(out)
    cb = color_balance_loss(out,target)
    gb = green_bias_loss(out, target)
    v = vibrance_loss(out, target)

    return (0.5*l1 + 0.1*l2 + 0.3*s + 0.2*p + 0.15*c + 0.05*k + 0.1*e +0.03*cb + 0.02*gb + 0.05*v)
