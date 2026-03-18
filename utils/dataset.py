import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class LOLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir should contain:
            high/
            low/
        """
        self.low_dir = os.path.join(root_dir, "low")
        self.high_dir = os.path.join(root_dir, "high")

        # self.filenames = sorted([
        #     f for f in os.listdir(self.low_dir)
        #     if f.endswith(".png") and os.path.exists(os.path.join(self.high_dir, f))
        # ])

        self.expert_dirs = [
        os.path.join(root_dir, "ExpertC"),
        ]

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        low_path = os.path.join(self.low_dir, filename)
        high_path = os.path.join(self.high_dir, filename)

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        low_img = self.transform(low_img)
        high_img = self.transform(high_img)

        return low_img, high_img
