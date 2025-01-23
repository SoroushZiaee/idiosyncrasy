from torch.utils.data import Dataset
import pandas as pd

from torchvision import transforms

import os
import torch
from PIL import Image


class Coco1600SequenceDataset(Dataset):
    def __init__(self, meta_file, img_dir, num_timesteps=5, transform=None):
        self.img_labels = pd.read_csv(meta_file)
        self.img_dir = img_dir
        self.num_timesteps = num_timesteps
        self.transform = transform
        self.to_grayscale = transforms.Grayscale(3)  # 3-channel grayscale

        # Create index map for object labels
        self.label_to_idx = {
            label: idx for idx, label in enumerate(self.img_labels["obj"].unique())
        }

    def __len__(self):
        return len(self.img_labels)

    def __coco1600_to_hvm200(self, label):

        coco1600_to_hvm200 = {
            "bear": "bear",
            "ELEPHANT_M": "elephant",
            "face0001": "person",
            "alfa155": "car",
            "breed_pug": "dog",
            "Apple_Fruit_obj": "apple",
            "_001": "chair",
            "f16": "plane",
            "lo_poly_animal_CHICKDEE": "bird",
            "zebra": "zebra",
        }

        return coco1600_to_hvm200[label]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_labels.iloc[idx]["image_names"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label = self.img_labels.iloc[idx]["obj"]
        label_idx = self.label_to_idx[label]

        # Create sequence: first frame is the original image, rest are grayscale
        sequence = [
            self.transform(image) if self.transform else transforms.ToTensor()(image)
            for _ in range(3)
        ]
        for _ in range(self.num_timesteps - 1):
            # gray_value = torch.rand(1).item()
            gray_value = 0.5

            gray_frame = torch.full(
                (3, 224, 224), gray_value
            )  # Assuming 224x224 is your image size
            sequence.append(gray_frame)

        sequence = torch.stack(sequence)
        return sequence, label_idx
