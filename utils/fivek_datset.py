import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset


class FiveKDataset(Dataset):
    def __init__(self, root_dir):

        self.input_dir = os.path.join(root_dir, "UneditedsRGB")

        self.expert_dirs = [
            # os.path.join(root_dir, "ExpertA"),
            # os.path.join(root_dir, "ExpertB"),
            os.path.join(root_dir, "ExpertC"),
            # os.path.join(root_dir, "ExpertD"),
            # os.path.join(root_dir, "ExpertE"),
        ]

        # ✅ LIMIT DATASET HERE (correct place)
        self.filenames = sorted(os.listdir(self.input_dir))

        self.samples = []

        for filename in self.filenames:
            for idx, expert_dir in enumerate(self.expert_dirs):
                self.samples.append((filename, expert_dir, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        filename, expert_dir, expert_idx = self.samples[idx]

        input_path = os.path.join(self.input_dir, filename)
        expert_path = os.path.join(expert_dir, filename)

        input_img = Image.open(input_path).convert("RGB")
        expert_img = Image.open(expert_path).convert("RGB")

        # SAME random crop
        i, j, h, w = transforms.RandomCrop.get_params(input_img, (256, 256))

        input_img = TF.crop(input_img, i, j, h, w)
        expert_img = TF.crop(expert_img, i, j, h, w)

        input_img = TF.to_tensor(input_img)
        expert_img = TF.to_tensor(expert_img)

        return input_img, expert_img, expert_idx