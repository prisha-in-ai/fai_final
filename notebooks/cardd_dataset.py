import os
from PIL import Image
from torch.utils.data import Dataset

class CarDDMultiLabelDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(".jpg")
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))
        labels = open(label_path).read().strip().split()
        labels = torch.tensor([float(x) for x in labels])

        if self.transform:
            image = self.transform(image)

        return image, labels
