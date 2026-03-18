from utils.fivek_datset import FiveKDataset
from torch.utils.data import DataLoader

dataset = FiveKDataset("dataset/OrignalDataset/Usable")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for inp, target in loader:
    print(inp.shape, target.shape)
    break
