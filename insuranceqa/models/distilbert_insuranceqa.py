import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, AutoConfig, AutoModelForSequenceClassification
from typing import Optional
import numpy as np
from pytorch_lightning.utilities import rank_zero_only


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        num_labels: int = 12,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 1e-10,
        eval_splits: Optional[list] = None,
        freeze_layers=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels, problem_type="multi_label_classification"
        )
        self.l1 = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

        if freeze_layers:
            for name, param in self.l1.named_parameters():
                if "distilbert" in name:
                    param.requires_grad = False

    def forward(self, **inputs):
        output_l1 = self.l1(**inputs)
        return output_l1

    @rank_zero_only
    def _log_metrics(self, key, values):
        value = np.mean(values)
        self.logger.experiment.log_metric(key=key, value=value, run_id=self.logger.run_id)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("loss", outputs.loss)
        return {"loss": outputs.loss}

    def training_epoch_end(self, outputs):
        all_preds = [output["loss"].cpu().numpy() for output in outputs]
        self._log_metrics("loss", all_preds)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        metric = {"val_loss": outputs.loss}
        self.log("val_loss", outputs.loss)
        return metric

    def validation_epoch_end(self, outputs):
        all_preds = [output["val_loss"].cpu().numpy() for output in outputs]
        self.log("val_loss", np.mean(all_preds))
        self._log_metrics("val_loss", all_preds)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self.model(**batch)
        return y_hat

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)
