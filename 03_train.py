# Databricks notebook source
# MAGIC %md This notebook can be found at https://github.com/databricks-industry-solutions/insurance-qa-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Insurance Q&A Intent Classification with Databricks & Hugging Face
# MAGIC ### Model Training
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rafaelvp-db/dbx-insurance-qa-hugging-face/master/img/header.png" width="800px"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Distilbert Example
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC * In the cell below, we simply download a pre-trained, distilled bersion of BERT (`distilbert-base-uncased`) to see how it works.
# MAGIC * We go on and run a sample prediction for a piece of text

# COMMAND ----------

# MAGIC %pip install transformers==4.24

# COMMAND ----------

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
  outputs = model(**inputs, labels = torch.tensor(0))
  logits = outputs.logits
  loss = outputs.loss

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
print(loss)

# COMMAND ----------

# DBTITLE 1,Model Architecture
model

# COMMAND ----------

# DBTITLE 1,Customizing our Model
# MAGIC %md
# MAGIC 
# MAGIC * Our model must have 12 different labels; we need to customize `DistilBERT` to support that
# MAGIC * But first, we need to have a Torch `Dataset` and a `DataLoader`. Let's code that up
# MAGIC * We will do this in a way that we directly query data that is stored in a Delta table

# COMMAND ----------

import torch
from torch.utils.data import Dataset, DataLoader
from pyspark.storagelevel import StorageLevel
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

class InsuranceDataset(Dataset):
    def __init__(
      self,
      database_name = "insuranceqa",
      split = "train",
      input_col = "question_en",
      label_col = "topic_en",
      storage_level = StorageLevel.MEMORY_ONLY,
      tokenizer = "distilbert-base-uncased",
      max_length = 512
    ):
      super().__init__()
      self.input_col = input_col
      self.label_col = label_col
      self.df = spark.sql(
        f"select * from {database_name}.{split}"
      ).toPandas()
      self.length = len(self.df)
      self.class_mappings = self._get_class_mappings()
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
      self.max_length = max_length

    def _get_class_mappings(self):
      labels = self.df \
        .topic_en \
        .unique()

      indexes = LabelEncoder().fit_transform(labels)
      class_mappings = dict(zip(labels, indexes))
      return class_mappings

    def _encode_label(self, label):
      self._get_class_mappings()
      label_class = self.class_mappings[label]
      encoded_label = torch.nn.functional.one_hot(
        torch.tensor([label_class]),
        num_classes = len(self.class_mappings)
      )
      return encoded_label.type(torch.float)

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
      question = self.df.loc[idx, self.input_col]
      inputs = self.tokenizer(
        question,
        None,
        add_special_tokens=True,
        return_token_type_ids=True,
        truncation=True,
        max_length=self.max_length,
        padding="max_length"
      )

      ids = inputs['input_ids']
      mask = inputs['attention_mask']
      labels = self.df.loc[idx, self.label_col]
      labels = self._encode_label(labels)[0]
      return {
        "input_ids": torch.tensor(ids, dtype = torch.long),
        "attention_mask": torch.tensor(mask, dtype = torch.long),
        "labels": labels
      }

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's see how our dataset samples look like:

# COMMAND ----------

training_data = InsuranceDataset(split = "train")
training_data[100]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Data Loaders
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC * PyTorch provides DataLoaders, which are a simple way to provide a batching interface on top of your datasets.
# MAGIC * They can be configured to work on a distributed fashion, by setting the `num_workers` parameter

# COMMAND ----------

# DBTITLE 1,Instantiating Data Loaders
from torch.utils.data import DataLoader

test_data = InsuranceDataset(split = "test")
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=20)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=20)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Lightning Modules
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC * Time to declare our Lightning Module!
# MAGIC * Lightning Module is quite a useful class provided by PyTorch Lightning. It helps us avoid a lot of boiler plate code - and also avoid forgetting about `torch.no_grad()` when running forward pass 😃
# MAGIC * We can customize how our model is going to be loaded, which layers in the model we are going to train or not, how metrics will be logged, etc.

# COMMAND ----------

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
        freeze_layers = True,
        **kwargs,
    ):
      super().__init__()
      self.save_hyperparameters()
      self.config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels = num_labels,
        problem_type = "multi_label_classification"
      )
      self.l1 = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config = self.config
      )

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
      self.logger.experiment.log_metric(
        key = key,
        value = value,
        run_id = self.logger.run_id
      )

    def training_step(self, batch, batch_idx):
      outputs = self(**batch)
      self.log("loss", outputs.loss)
      return {"loss": outputs.loss}

    def training_epoch_end(self, outputs):
      all_preds = [output["loss"].cpu().numpy() for output in outputs]
      self._log_metrics("loss", all_preds)

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
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

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Training our Model
# MAGIC 
# MAGIC Here we will:
# MAGIC 
# MAGIC * Declare an `EarlyStoppingCallback` - this way we can avoid overfitting and don't need to think about number of training epochs
# MAGIC * Declare an `MLFlowLogger` - since MLFlow `autologging` doesn't work for PyTorch Lightning > 1.5
# MAGIC * Create a `Trainer` to train our model
# MAGIC * Train our model!

# COMMAND ----------

import mlflow
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

early_stop_callback = EarlyStopping(
  monitor="val_loss",
  min_delta=0.001,
  patience=3,
  verbose=True,
  mode="min"
)

with mlflow.start_run(run_name = "torch") as run:
  lit_model = LitModel()
  run_id = run.info.run_id
  experiment_name = mlflow.get_experiment(run.info.experiment_id)

  mlf_logger = MLFlowLogger(
      experiment_name = experiment_name,
      run_id=run_id,
      tracking_uri="databricks"
  )

  trainer = pl.Trainer(
    max_epochs = 10000,
    logger = mlf_logger,
    accelerator="gpu",
    devices = 4,
    callbacks = [early_stop_callback]
  )

  trainer.fit(
    lit_model,
    train_dataloaders = train_dataloader,
    val_dataloaders = test_dataloader
  )

# COMMAND ----------

def predict(question = "how do i get car insurance?", ground_truth = "auto-insurance"):

  encodings = tokenizer(
    question,
    None,
    add_special_tokens=True,
    return_token_type_ids=True,
    truncation=True,
    max_length=512,
    padding="max_length"
  )

  inputs = {
    "input_ids": torch.tensor([encodings["input_ids"]], dtype = torch.long),
    "attention_mask": torch.tensor([encodings["attention_mask"]], dtype = torch.long)
  }

  with torch.no_grad():
    y_hat = lit_model(**inputs)

  label = y_hat.logits.argmax().item()
  return label

# COMMAND ----------

# DBTITLE 1,Testing our model
for i in range(0, 10):
  sample = test_data.df.sample(1)
  question = sample["question_en"].values[0]
  topic = sample["topic_en"].values[0]

  pred = predict(question, topic)
  print(f"Question: {question}")
  print(f"Predicted '{topic}'? {pred == training_data.class_mappings[topic]}")
  print(f"------------------------------------------------------------------")

# COMMAND ----------

# DBTITLE 1,Logging our Artifact and Registering our Model
with mlflow.start_run(run_id = run.info.run_id) as run:

  mlflow.pytorch.log_model(
    lit_model,
    artifact_path = "model",
    registered_model_name = "distilbert_en_uncased_insuranceqa"
  )

# COMMAND ----------

class_mappings = {}

for key in training_data.class_mappings.keys():
  class_mappings[key] = training_data.class_mappings[key]

class_mappings

# COMMAND ----------

# DBTITLE 1,A bit of free form testing with (completely) unseen data
test_questions = {
  "my car broke, what should I do?": 1,
  "my mother is sick, what should I do?": 4,
  "can I talk about my pension?": 11,
  "I've been deemed incapacitated for work, how do I make a claim?": 3,
  "my wife wants to know what's in our life policy": 6,
  "can I ask a question on medicare?": 8,
  "what's the coverage for medicare?": 8,
  "which retirement plans you have?": 11,
  "how you invest money for retirement plan?": 11,
  "I'm 60, what's my health premium?": 4
}

for question, topic in test_questions.items():

  pred = predict(question, topic)
  print(f"Question: {question}")
  print(f"Prediction: {pred}")
  print(f"Predicted '{topic}'? {pred == topic}")
  print(f"------------------------------------------------------------------")

