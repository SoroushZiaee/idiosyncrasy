import os
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from cka_reg.datasets._ImageNetDataset import ImageNet
from cka_reg import IMAGENET_PATH

class ImagenetDataModule(LightningDataModule):
    
    name = "ImageNet"
    
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.hparams.update(vars(hparams))

        self.data_dir = IMAGENET_PATH
        self.image_size = self.hparams.image_size
        self.batch_size = self.hparams.batch_size
        print(f"*"*100)
        print(f"\nBatch size: {self.batch_size}")
        print(f"*"*100)
        
        self.num_workers = self.hparams.num_workers
        self.temp_extract = self.hparams.temp_extract
        self.pin_memory_train, self.pin_memory_val, self.pin_memory_test = (
            self.hparams.pin_memories
        )

        self.dims = (3, self.image_size, self.image_size)
        self.num_classes = 1000  # ImageNet has 1000 classes
        self.task_type = "classification"  # Classification task

    def prepare_data(self):
        # Check if the data is already downloaded
        if not os.path.exists(
            os.path.join(self.data_dir, "train")
        ) or not os.path.exists(os.path.join(self.data_dir, "val")):
            print("ImageNet data not found. Please download the dataset manually.")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.get_dataset("train", self.train_transform())
            self.val_dataset = self.get_dataset("val", self.val_transform())

        if stage == "test" or stage is None:
            self.test_dataset = self.get_dataset("val", self.val_transform())

        if stage == "fit" or stage is None:
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")

    def get_dataset(self, split: str, transform):
        return ImageNet(
            root=self.data_dir,
            split=split,
            temp_extract=self.temp_extract,
            transform=transform,
        )

    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def val_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory_train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory_val,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory_test,
        )

    def log_samples_to_tensorboard(self, logger):
        if self.task_type == "classification" or self.task_type == "combined":
            # Get a batch of data
            batch = next(iter(self.train_dataloader()))
            images, labels = batch
            if self.task_type == "combined":
                images, labels = images["classification"], labels["classification"]

            # Create a grid of images
            grid = torchvision.utils.make_grid(images)
            logger.experiment.add_image("sample_images", grid, 0)

            # Log labels
            if self.task_type == "classification":
                class_names = [f"Class_{i}" for i in range(self.num_classes)]
                # label_names = [class_names[label] for label in labels]
                logger.experiment.add_text("sample_labels", ", ".join(class_names), 0)
            elif self.task_type == "combined":
                logger.experiment.add_text(
                    "sample_classification_labels", str(labels.tolist()), 0
                )

        if self.task_type == "regression" or self.task_type == "combined":
            batch = next(iter(self.train_dataloader()))
            images, labels = batch
            if self.task_type == "combined":
                images, labels = images["regression"], labels["regression"]

            # Create a grid of images
            grid = torchvision.utils.make_grid(images)
            logger.experiment.add_image("sample_regression_images", grid, 0)

            # Log labels
            logger.experiment.add_text(
                "sample_regression_labels", str(labels.tolist()), 0
            )
            
# class ImagenetDataModule(LightningDataModule):

#     name = "ImageNet"

#     def __init__(self, hparams, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # self.hparams = hparams
#         self.hparams.update(vars(hparams))
#         self.image_size = hparams.image_size
#         self.dims = (3, self.image_size, self.image_size)
#         self.root = IMAGENET_PATH
#         self.meta_dir = None
#         self.num_workers = hparams.num_workers
#         self.batch_size = hparams.batch_size
#         self.temp_extract = hparams.temp_extract

#     def _get_dataset(self, type_, transforms):

#         dataset = ImageNet(
#             root=self.root,
#             split=type_,
#             temp_extract=self.temp_extract, # hparams.temp_extract, # True
#             # dst_meta_path=self.meta_dir,
#             transform=transforms,
#         )
#         dataset.name = self.name
#         return dataset

#     def _get_DataLoader(self, *args, **kwargs):
#         return DataLoader(*args, **kwargs)

#     def train_transform(self):
#         """
#         The standard imagenet transforms
#         """
#         preprocessing = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(self.image_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 # imagenet_normalization(), # model does it's own normalization!
#             ]
#         )

#         return preprocessing

#     def val_transform(self):
#         """
#         The standard imagenet transforms for validation
#         """
#         preprocessing = transforms.Compose(
#             [
#                 transforms.Resize(self.image_size + 32),
#                 transforms.CenterCrop(self.image_size),
#                 transforms.ToTensor(),
#                 # imagenet_normalization(), # model does it's own normalization!
#             ]
#         )
#         return preprocessing

#     def train_dataloader(self):
#         transforms = self.train_transform()
#         dataset = self._get_dataset("train", transforms)

#         loader = self._get_DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             drop_last=True,
#             pin_memory=True,
#         )
#         return loader

#     def val_dataloader(self):
#         """
#         Uses the validation split of imagenet2012 for testing
#         """
#         transforms = self.val_transform()
#         dataset = self._get_dataset("val", transforms)
#         loader = self._get_DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             drop_last=True,
#             pin_memory=True,
#         )
#         return loader
