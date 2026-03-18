import os
import shutil
from PIL import Image
import re

source_dir = r"C:\Users\sarth\Downloads\Data"
target_root = r"C:\Users\sarth\Desktop\FreshStart\dataset\OrignalDataset\Usable"

experts = ["ExpertA", "ExpertB", "ExpertC", "ExpertD", "ExpertE", "UneditedsRGB"]

# Create folders
for exp in experts:
    os.makedirs(os.path.join(target_root, exp), exist_ok=True)


for filename in os.listdir(source_dir):

    filepath = os.path.join(source_dir, filename)

    if filename.endswith(".tif"):

        # Extract base name (e.g., a0002)
        base_match = re.match(r"(a\d+)", filename)
        if not base_match:
            continue

        base_name = base_match.group(1) + ".png"

        # Determine expert type
        if "(1)" in filename:
            expert_folder = "ExpertA"
        elif "(2)" in filename:
            expert_folder = "ExpertB"
        elif "(3)" in filename:
            expert_folder = "ExpertC"
        elif "(4)" in filename:
            expert_folder = "ExpertD"
        else:
            expert_folder = "ExpertE"

        # Convert to PNG
        img = Image.open(filepath).convert("RGB")
        save_path = os.path.join(target_root, expert_folder, base_name)
        img.save(save_path)

        print(f"Saved {base_name} to {expert_folder}")

    elif filename.endswith(".dng"):

        # RAW → skip for now or handle later
        base_match = re.match(r"(a\d+)", filename)
        if not base_match:
            continue

        base_name = base_match.group(1) + ".png"

        print(f"RAW found: {filename} — convert manually to PNG first.")
