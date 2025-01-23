from argparse import Namespace
import lightning as l
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from torchmetrics import Accuracy, F1Score
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

from nn.models.working_memory import ResNetLSTMFeedback, ResNetLSTM


class ResNetLSTMModule(l.LightningModule):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = Path(config_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Convert string values to appropriate types
        self.example_input_array = torch.randn(
            1, self.config["data"]["num_timesteps"], 3, 224, 224
        )
        self.model_config = self._process_model_config(self.config["model"])

        # Process all configurations
        self.processed_config = self._process_all_configs(self.config)

        # Initialize the selected model
        self.model = self._initialize_model()

        # Initialize metrics
        self.train_metrics = torch.nn.ModuleDict(
            {
                "acc": Accuracy(
                    task="multiclass", num_classes=self.model_config["num_classes"]
                ),
                "f1": F1Score(
                    task="multiclass", num_classes=self.model_config["num_classes"]
                ),
            }
        )

        self.val_metrics = torch.nn.ModuleDict(
            {
                "acc": Accuracy(
                    task="multiclass", num_classes=self.model_config["num_classes"]
                ),
                "f1": F1Score(
                    task="multiclass", num_classes=self.model_config["num_classes"]
                ),
            }
        )

        # Save hyperparameters for logging
        self.save_hyperparameters()

    def _initialize_model(self):
        """Initialize the selected model based on configuration."""
        model_type = self.model_config.get("model_type", "resnet_lstm_feedback")

        model_classes = {
            "resnet_lstm_feedback": ResNetLSTMFeedback,
            "resnet_lstm": ResNetLSTM,
        }

        if model_type not in model_classes:
            raise ValueError(
                f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}"
            )

        selected_model = model_classes[model_type](
            num_classes=self.model_config["num_classes"],
            hidden_size=self.model_config["hidden_size"],
            num_layers=self.model_config["num_layers"],
            dropout_rate=self.model_config["dropout_rate"],
        )

        logger.info(f"Initialized model type: {model_type}")
        return selected_model

    def _process_all_configs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process all configuration sections."""
        processed = {}

        # Process model config
        processed["model"] = self._process_model_config(config.get("model", {}))

        # Process training config
        processed["training"] = {
            "max_epochs": int(config.get("training", {}).get("max_epochs", 150)),
            "precision": int(config.get("training", {}).get("precision", 32)),
            "accumulate_grad_batches": int(
                config.get("training", {}).get("accumulate_grad_batches", 1)
            ),
            "gradient_clip_val": float(
                config.get("training", {}).get("gradient_clip_val", 0.0)
            ),
            "early_stopping_patience": int(
                config.get("training", {}).get("early_stopping_patience", 10)
            ),
        }

        # Process data config
        processed["data"] = {
            "batch_size": int(config.get("data", {}).get("batch_size", 32)),
            "num_workers": int(config.get("data", {}).get("num_workers", 4)),
            "num_timesteps": int(config.get("data", {}).get("num_timesteps", 5)),
            "train_val_split": float(
                config.get("data", {}).get("train_val_split", 0.8)
            ),
        }

        # Process transform config
        processed["transforms"] = {
            "resize_size": config.get("transforms", {}).get("resize_size", [224, 224]),
            "normalize_mean": config.get("transforms", {}).get(
                "normalize_mean", [0, 0, 0]
            ),
            "normalize_std": config.get("transforms", {}).get(
                "normalize_std", [1, 1, 1]
            ),
        }

        return processed

    def _process_model_config(self, config):
        """Convert config values to appropriate types."""
        processed_config = {}
        processed_config["model_type"] = str(
            config.get("model_type", "resnet_lstm_feedback")
        )
        processed_config["num_classes"] = int(config["num_classes"])
        processed_config["hidden_size"] = int(config["hidden_size"])
        processed_config["num_layers"] = int(config["num_layers"])
        processed_config["dropout_rate"] = float(config["dropout_rate"])
        processed_config["learning_rate"] = float(config["learning_rate"])
        processed_config["weight_decay"] = float(config["weight_decay"])
        return processed_config

    def _get_model_size(self) -> Dict[str, int]:
        """Calculate model size statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }

    def _log_hyperparameters(self):
        """Log hyperparameters to TensorBoard."""
        if self.logger is not None and hasattr(self.logger, "log_hyperparams"):
            # Combine all hyperparameters

            # Log to TensorBoard
            self.logger.log_hyperparams(self.hparams)

            # Log config file content as text
            with open(self.config_path) as f:
                config_content = f.read()
                self.logger.experiment.add_text(
                    "config/raw", f"```yaml\n{config_content}\n```"
                )

    def on_fit_start(self):
        """Called when fit begins."""
        if self.logger is not None:
            # Log model summary
            self.logger.log_hyperparams(
                Namespace(**self.config),
                metrics={
                    "hp_metric": self.trainer.callback_metrics.get(
                        self.config["training"]["monitor_metric"], 0
                    )
                },
            )

            model_info = str(self.model)
            self.logger.experiment.add_text("model/summary", f"```\n{model_info}\n```")

            # Log additional information about the training setup
            setup_info = (
                f"Training Device: {self.device}\n"
                f"Precision: {self.trainer.precision}\n"
                f"Gradient Accumulation Steps: {self.trainer.accumulate_grad_batches}\n"
                f"Max Epochs: {self.trainer.max_epochs}\n"
            )
            self.logger.experiment.add_text("training/setup", setup_info)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        outputs = self(sequences)
        loss = F.cross_entropy(outputs, labels)

        # Calculate predictions
        preds = torch.argmax(outputs, dim=1)

        # Update metrics
        for name, metric in self.train_metrics.items():
            value = metric(preds, labels)
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(labels),
            )

        # Log loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labels),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        outputs = self(sequences)
        loss = F.cross_entropy(outputs, labels)

        # Calculate predictions
        preds = torch.argmax(outputs, dim=1)

        # Update metrics
        for name, metric in self.val_metrics.items():
            value = metric(preds, labels)
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(labels),
            )

        # Log loss
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labels),
        )

        return {"val/loss": loss}

    def on_train_epoch_start(self):
        # Reset metrics at the start of each epoch
        for metric in self.train_metrics.values():
            metric.reset()

    def on_validation_epoch_start(self):
        # Reset metrics at the start of each epoch
        for metric in self.val_metrics.values():
            metric.reset()

    def on_train_epoch_end(self):
        # Compute final metrics for the epoch
        for name, metric in self.train_metrics.items():
            value = metric.compute()  # compute final value
            self.log(f"train/epoch_{name}", value, on_epoch=True)

    def on_validation_epoch_end(self):
        # Compute final metrics for the epoch
        for name, metric in self.val_metrics.items():
            value = metric.compute()  # compute final value
            self.log(f"val/epoch_{name}", value, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model_config["learning_rate"],
            weight_decay=self.model_config["weight_decay"],
            betas=(0.9, 0.999),
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
