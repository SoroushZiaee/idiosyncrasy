import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from cka_reg.datasets import ImageNet
from cka_reg import IMAGENET_PATH


class ImagenetDataModule(LightningDataModule):

    name = "ImageNet"

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.image_size = hparams.image_size
        self.dims = (3, self.image_size, self.image_size)
        self.root = IMAGENET_PATH
        self.meta_dir = os.path.expanduser(
            "~/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet"
        )
        self.num_workers = hparams.num_workers
        self.batch_size = hparams.batch_size

    def _get_dataset(self, type_, transforms):

        dataset = ImageNet(
            root=self.root,
            split=type_,
            dst_meta_path=self.meta_dir,
            transform=transforms,
        )
        dataset.name = self.name
        return dataset

    def _get_DataLoader(self, *args, **kwargs):
        return DataLoader(*args, **kwargs)

    def train_transform(self):
        """
        The standard imagenet transforms
        """
        preprocessing = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # imagenet_normalization(), # model does it's own normalization!
            ]
        )

        return preprocessing

    def val_transform(self):
        """
        The standard imagenet transforms for validation
        """
        preprocessing = transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                # imagenet_normalization(), # model does it's own normalization!
            ]
        )
        return preprocessing

    def train_dataloader(self):
        transforms = self.train_transform()
        dataset = self._get_dataset("train", transforms)

        loader = self._get_DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """
        Uses the validation split of imagenet2012 for testing
        """
        transforms = self.val_transform()
        dataset = self._get_dataset("validation", transforms)
        loader = self._get_DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader
