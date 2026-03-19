import os
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF

from models.crisp import CRISP

device = "cuda"

root = "dataset/OrignalDataset/Usable"
input_path = os.path.join(root, "UneditedsRGB", "a0011.png")

expert_folders = ["ExpertA", "ExpertB", "ExpertC", "ExpertD", "ExpertE"]

# Load model
model = CRISP().to(device)

#--------------------------------------------------------------------------------------
# model.load_state_dict(
#     torch.load("crisp_fivek_100samples_90ep.pth", map_location=device)
# )
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
checkpoint = torch.load("crisp_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
#--------------------------------------------------------------------------------------

model.eval()

# Load input image
input_img = Image.open(input_path).convert("RGB")
input_tensor = TF.to_tensor(input_img).unsqueeze(0).to(device)

os.makedirs("results_fivek", exist_ok=True)

with torch.no_grad():

#-------------------------- 5 Expert-----------------------------------
    '''
    for idx, expert_name in enumerate(expert_folders):

        style = torch.zeros(1, 5).to(device)
        style[0, idx] = 1.0

        phi = model.decoder(style)
        phi = 0.3 * phi   # important (same as training)

        output = model.isp(input_tensor, phi)

        save_image(output, f"results_fivek/output_{expert_name}.png")

        print("Saved style:", expert_name)
    '''
#-------------------------- 5 Expert-----------------------------------

    
#-------------------------- 1 Expert-----------------------------------

    style = torch.zeros(1, 5).to(device)
    style[0, 0] = 1.0

    phi = model.decoder(style)
    phi = 0.6 * phi  

    output = model.isp(input_tensor, phi)

    save_image(output, f"results_fivek/output_ExpertC.png")

    print("Saved style:", "ExpertC")

#-------------------------- 1 Expert-----------------------------------