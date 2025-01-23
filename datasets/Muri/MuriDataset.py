import os
from torch.utils.data import Dataset
from torchvision import transforms as T
import PIL.Image
import re
import pandas as pd
from glob import glob
import torch


class MuriDataset(Dataset):
    def __init__(self, root: str, transforms=None, return_paths=False):

        self.root = root

        self.transforms = transforms
        img_path_list = os.listdir(root)
        img_path_list.sort()

        self.img_path_list = img_path_list
        self.return_path = return_paths

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.img_path_list[idx])
        # print(f"{img_name = }")
        image = PIL.Image.open(img_name).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        if self.return_path:
            return image, os.path.basename(img_name)

        return image


class Muri1320SequenceDataset(Dataset):
    def __init__(self, image_path, num_timesteps=5, transform=None):
        self.image_path = image_path
        self.num_timesteps = num_timesteps
        self.transform = transform or T.Compose([T.Resize((224, 224)), T.ToTensor()])

        # Sort image files
        self.images = sort_filenames(glob(os.path.join(image_path, "*.png")))

        # Load metadata
        meta_data_path = os.path.join(image_path, "working_memory_images_labels.csv")
        self.meta_data = pd.read_csv(meta_data_path)
        self.meta_data["img_path"] = self.images

        # Create a mapping of unique objects to integer labels
        self.label_to_idx = {
            obj: i for i, obj in enumerate(self.meta_data["object"].unique())
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        image = PIL.Image.open(img_path).convert("RGB")

        # Create sequence: first frame is the original image, rest are grayscale
        sequence = [self.transform(image) for _ in range(3)]
        for _ in range(self.num_timesteps - 1):
            gray_value = 0.5
            gray_frame = torch.full((3, 224, 224), gray_value)
            sequence.append(gray_frame)

        sequence = torch.stack(sequence)

        label = self.meta_data.loc[
            self.meta_data["img_path"] == img_path, "object"
        ].values[0]
        label_idx = self.label_to_idx[label]

        return sequence, label_idx

    def hvm200_to_coco1600(self, label):
        hvm200_to_coco1600 = {
            "bear": "bear",
            "elephant": "ELEPHANT_M",
            "person": "face0001",
            "car": "alfa155",
            "dog": "breed_pug",
            "apple": "Apple_Fruit_obj",
            "chair": "_001",
            "plane": "f16",
            "bird": "lo_poly_animal_CHICKDEE",
            "zebra": "zebra",
        }
        return hvm200_to_coco1600.get(label, label)


def extract_number(filename):
    # Extract the number from the filename
    match = re.search(r"im(\d+)\.png", filename)
    if match:
        return int(match.group(1))
    return 0  # Return 0 if no number is found


def sort_filenames(filenames):
    # Sort the filenames based on the extracted number
    return sorted(filenames, key=extract_number)
