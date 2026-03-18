import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils.dataset import LOLDataset
from models.crisp import CRISP

device = "cuda"

# Load dataset (eval set)
dataset = LOLDataset("dataset/LOL_eval15")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load model
model = CRISP().to(device)
model.load_state_dict(torch.load("crisp_model.pth"))
model.eval()

with torch.no_grad():
    for i, (low, high) in enumerate(loader):

        low = low.to(device)
        high = high.to(device)

        output, _ = model(low, high)

        save_image(low, f"results/{i}_low.png")
        save_image(high, f"results/{i}_high.png")
        save_image(output, f"results/{i}_output.png")

        if i == 4:
            break
