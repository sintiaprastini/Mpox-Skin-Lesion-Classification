import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

CLASSES = ['Chickenpox', 'Cowpox', 'Healthy', 'HFMD', 'Measles', 'Monkeypox']

class getImageLabel(Dataset):
    def __init__(self, folders, img_size=64):
        if isinstance(folders, str):
            folders = [folders]

        self.features = []
        self.labels = []
        to_one_hot = np.eye(len(CLASSES), dtype=np.float32)

        for folder in folders:
            for i, cls_name in enumerate(CLASSES):
                cls_dir = os.path.join(folder, cls_name)
                if not os.path.exists(cls_dir):
                    continue
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    try:
                        # Pre-processing: RGB conversion, Autocontrast, Resize, Normalization (0-1)
                        img = Image.open(img_path).convert('RGB')
                        img = ImageOps.autocontrast(img)
                        img = img.resize((img_size, img_size))
                        arr = np.asarray(img, dtype=np.float32) / 255.0
                        # Ubah dimensi dari (H, W, C) ke (C, H, W)
                        arr = np.transpose(arr, (2, 0, 1))
                        self.features.append(arr)
                        self.labels.append(to_one_hot[i])
                    except Exception:
                        continue

        self.features = torch.tensor(np.array(self.features), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

    def __len__(self):
        return len(self.features)
