import os
import rawpy
import imageio
import re

source_dir = r"C:\Users\sarth\Downloads\Data"
target_dir = r"C:\Users\sarth\Desktop\FreshStart\dataset\OrignalDataset\Usable\UneditedsRGB"

os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(source_dir):

    if filename.endswith(".dng"):

        filepath = os.path.join(source_dir, filename)

        # Extract base name like a0002
        match = re.match(r"(a\d+)", filename)
        if not match:
            continue

        base_name = match.group(1) + ".png"
        save_path = os.path.join(target_dir, base_name)

        with rawpy.imread(filepath) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8
            )

        imageio.imwrite(save_path, rgb)

        print(f"Converted: {filename} → {base_name}")
