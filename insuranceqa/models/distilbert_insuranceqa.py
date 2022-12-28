import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification
from typing import Optional
import numpy as np
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import AdamW

class LitModel(pl.LightningModule):
    def __init__(
      self,
        loss,
        model_name_or_path: str = "distilbert-base-uncased",
        num_labels: int = 12,
        learning_rate: float = 1e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 1e-10,
        eval_splits: Optional[list] = None,
        freeze_layers = True,
        **kwargs,
    ):
      super().__init__()
      self.save_hyperparameters(ignore=["loss"])
      self.learning_rate = learning_rate
      self.config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels = num_labels,
        problem_type = "multi_label_classification"
      )
      self.model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config = self.config
      )

      if freeze_layers:
        for name, param in self.model.named_parameters():
          if name not in ["classifier.weight", "classifier.bias"]:
              param.requires_grad = False

      if loss:
        self.loss = loss
      else: 
        self.loss = nn.CrossEntropyLoss()

    def forward(self, **inputs):
      output = self.model(**inputs)
      return output

    def training_step(self, batch, batch_idx):
      outputs = self(**batch)
      loss = self.loss(outputs.logits, batch["labels"])
      self.logger.experiment.log_metric(
        key = "loss",
        value = loss,
        run_id = self.logger.run_id
      )
      return {"loss": loss}

    def training_epoch_end(self, outputs):
      losses = [output["loss"].cpu().numpy() for output in outputs]
      loss = np.mean(losses)
      self.logger.experiment.log_metric(
        key = "loss",
        value = loss,
        run_id = self.logger.run_id
      )

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
      outputs = self(**batch)
      loss = self.loss(outputs.logits, batch["labels"])
      metric = {"val_loss": loss}
      self.logger.experiment.log_metric(
        key = "val_loss",
        value = loss,
        run_id = self.logger.run_id
      )
      return metric

    def validation_epoch_end(self, outputs):
      losses = [output["val_loss"].cpu().numpy() for output in outputs]
      loss = np.mean(losses)
      self.logger.experiment.log_metric(
        key = "val_loss",
        value = loss,
        run_id = self.logger.run_id
      )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
      y_hat = self.model(**batch)
      return y_hat

    def configure_optimizers(self):
      return AdamW(self.parameters(), lr = self.learning_rate)