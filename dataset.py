from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from config import resize_x, resize_y
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class TrafficSignDataset(Dataset):
    def __init__(self, data_dir = 'data', transform=transform):
        self.images, self.labels = [], []
        for label, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def unicornLoader(dataset = TrafficSignDataset(), shuffle=True):
    from config import batchsize
    return DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
