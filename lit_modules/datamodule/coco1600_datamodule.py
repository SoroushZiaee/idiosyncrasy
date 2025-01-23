# data/data_module.py
from argparse import Namespace
import lightning as l
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from typing import Optional
import yaml
import torch
import torchvision.utils as vutils
from datasets.Coco.CocoDataset import Coco1600SequenceDataset
import logging
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)


class Coco1600DataModule(l.LightningDataModule):
    def __init__(self, config_path: str, logger=None):
        super().__init__()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.logger = logger
        self.logger.log_hyperparams(Namespace(**self.config))
        self.data_config = self.config["data"]
        self.transform_config = self.config["transforms"]
        self.setup_transforms()

        # Number of images to log
        self.num_viz_samples = 8

    def setup_transforms(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.transform_config["resize_size"]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Grayscale(3),
                transforms.Normalize(
                    mean=self.transform_config["normalize_mean"],
                    std=self.transform_config["normalize_std"],
                ),
            ]
        )

        # Base transform for visualization (always apply resize first)
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(self.transform_config["resize_size"]),
                transforms.ToTensor(),
                transforms.Grayscale(3),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize(self.transform_config["resize_size"]),
                transforms.ToTensor(),
                transforms.Grayscale(3),
                transforms.Normalize(
                    mean=self.transform_config["normalize_mean"],
                    std=self.transform_config["normalize_std"],
                ),
            ]
        )

    def denormalize(self, tensor):
        """Denormalize the tensor for visualization."""
        mean = torch.tensor(self.transform_config["normalize_mean"]).view(3, 1, 1)
        std = torch.tensor(self.transform_config["normalize_std"]).view(3, 1, 1)
        return tensor * std + mean

    def log_batch_visualization(
        self, logger, batch: torch.Tensor, set_name: str = "train"
    ):
        """Log a batch of images to TensorBoard."""
        try:
            sequences, labels = batch
            batch_size, timesteps, C, H, W = sequences.shape
            num_samples = min(self.num_viz_samples, batch_size)

            # Denormalize sequences for visualization
            sequences = sequences[:num_samples]
            sequences = sequences.view(-1, C, H, W)
            sequences = self.denormalize(sequences)

            # Create grid of images
            grid = vutils.make_grid(
                sequences, nrow=timesteps, normalize=True, value_range=(0, 1)
            )

            # Log to TensorBoard
            logger.experiment.add_image(
                f"samples/{set_name}_batch", grid, global_step=0
            )

            # Log individual sequences
            for i in range(num_samples):
                sequence = sequences[i * timesteps : (i + 1) * timesteps]
                grid_single = vutils.make_grid(
                    sequence, nrow=timesteps, normalize=True, value_range=(0, 1)
                )
                logger.experiment.add_image(
                    f"samples/{set_name}_sequence_{i}_label_{labels[i].item()}",
                    grid_single,
                    global_step=0,
                )

        except Exception as e:
            logger.error(f"Failed to log batch visualization: {str(e)}")

    def log_augmentation_examples(self, logger):
        """Log examples of data augmentation."""
        try:
            # Get a single sample with base transform
            dataset = Coco1600SequenceDataset(
                meta_file=self.data_config["meta_file"],
                img_dir=self.data_config["img_dir"],
                num_timesteps=self.data_config["num_timesteps"],
                transform=self.base_transform,  # Use base transform that includes resize
            )

            # Create augmentation transforms (applied after resize)
            aug_transforms = {
                "Original": transforms.Compose([]),
                "Flipped": transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p=1.0),
                    ]
                ),
                "Rotated": transforms.Compose(
                    [
                        transforms.RandomRotation(15),
                    ]
                ),
                "Color Jittered": transforms.Compose(
                    [
                        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    ]
                ),
            }

            # Get samples and apply augmentations
            for i in range(min(4, len(dataset))):
                try:
                    original_sequence, label = dataset[i]

                    # Apply each augmentation to the first frame only
                    first_frame = original_sequence[0]  # Get first frame
                    aug_frames = []

                    for aug_name, aug_transform in aug_transforms.items():
                        # Convert to PIL for transforms
                        frame_pil = transforms.ToPILImage()(first_frame)
                        # Apply augmentation
                        aug_frame = aug_transform(frame_pil)
                        # Convert back to tensor
                        aug_frame = transforms.ToTensor()(aug_frame)
                        aug_frames.append(aug_frame)

                    # Create and log grid
                    grid = vutils.make_grid(
                        torch.stack(aug_frames),
                        nrow=len(aug_transforms),
                        normalize=True,
                        value_range=(0, 1),
                    )

                    # Log to TensorBoard
                    logger.experiment.add_image(
                        f"augmentations/sample_{i}_label_{label}", grid, global_step=0
                    )

                    # Add text description
                    logger.experiment.add_text(
                        f"augmentations/sample_{i}_description",
                        " | ".join(aug_transforms.keys()),
                    )

                except Exception as e:
                    logger.error(f"Failed to process sample {i}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Failed to log augmentation examples: {str(e)}")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_dataset = Coco1600SequenceDataset(
                meta_file=self.data_config["meta_file"],
                img_dir=self.data_config["img_dir"],
                num_timesteps=self.data_config["num_timesteps"],
                transform=self.transform,
            )

            train_size = int(len(full_dataset) * self.data_config["train_val_split"])
            val_size = len(full_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

            # Override transform for validation dataset
            self.val_dataset.dataset.transform = self.val_transform

            # Log sample batches if logger is available
            if hasattr(self, "trainer") and self.trainer and self.trainer.logger:
                try:
                    # Log training samples
                    train_samples = next(iter(self.train_dataloader()))
                    self.log_batch_visualization(
                        self.trainer.logger, train_samples, "train"
                    )

                    # Log validation samples
                    val_samples = next(iter(self.val_dataloader()))
                    self.log_batch_visualization(
                        self.trainer.logger, val_samples, "val"
                    )

                    # Log augmentation examples
                    self.log_augmentation_examples(self.trainer.logger)
                except Exception as e:
                    logger.error(f"Failed to log visualizations during setup: {str(e)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config["batch_size"],
            num_workers=self.data_config["num_workers"],
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config["batch_size"],
            num_workers=self.data_config["num_workers"],
            shuffle=False,
            pin_memory=True,
        )
