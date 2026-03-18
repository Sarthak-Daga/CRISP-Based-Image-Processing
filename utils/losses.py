import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

def total_loss(out, target):
    l1 = F.l1_loss(out, target)
    p = perceptual_loss(out, target)
    c = color_loss(out, target)
    return l1 + 0.1 * p + 0.15 * c