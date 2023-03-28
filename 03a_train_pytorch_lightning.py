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

!pip install -q --upgrade pip && pip install -q pytorch_lightning==1.8.6 transformers

# COMMAND ----------

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

sentiments = {
  "LABEL_0": "Positive",
  "LABEL_1": "Negative"
}

sentence = "I'm happy today!"
inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
  outputs = model(**inputs, labels = torch.tensor(0))
  logits = outputs.logits
  loss = outputs.loss

predicted_class_id = logits.argmax().item()
label = model.config.id2label[predicted_class_id]
sentiment = sentiments[label]

print("\n****************************************************************************************")
print(f"Sentiment for the sentence: '{sentence}' was classified as {label} / {sentiment}")
print("****************************************************************************************")

outputs

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
import itertools
import collections

class InsuranceDataset(Dataset):
    def __init__(
      self,
      database_name = "insuranceqa",
      split = "train",
      input_col = "question_en",
      label_col = "topic_en",
      storage_level = StorageLevel.MEMORY_ONLY,
      tokenizer = "distilbert-base-uncased",
      max_length = 64
    ):
      super().__init__()
      self.input_col = input_col
      self.label_col = label_col
      self.split = split
      self.df = spark.sql(
        f"select * from {database_name}.{self.split}"
      ).toPandas()
      self.length = len(self.df)
      self.class_mappings = self._get_class_mappings()
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
      self.max_length = max_length
      self.weights = None

    def _get_class_mappings(self):
      labels = self.df \
        .topic_en \
        .unique()

      indexes = LabelEncoder().fit_transform(labels)
      class_mappings = dict(zip(labels, indexes))
      return class_mappings

    def _get_class_weights(self):

      # Calculating class_weights
      if self.split == "train":
        weights_df = self.df.groupby("topic_en").count().reset_index()
        weights_df["weight"] = 1 / weights_df["id"]
        self.weights = torch.tensor(weights_df["weight"].values)

      class_mappings = training_data.class_mappings
      weights = [
        (class_mappings[row[1]], row[0])
        for row in list(
          itertools.chain(spark.sql(query).toLocalIterator())
        )
      ]
      weights_dict = collections.OrderedDict(sorted(dict(weights).items()))
      weights_tensor = torch.tensor([weight[1] for weight in weights_dict.items()])
      self.weights = weights_tensor

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
training_data[0]

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

# Testing purposes; increase batch size when not testing
train_dataloader = DataLoader(training_data, batch_size=512, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

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

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Training our Model
# MAGIC 
# MAGIC Here we will:
# MAGIC 
# MAGIC * Declare an `EarlyStoppingCallback` - this way we can avoid overfitting and don't need to think about number of training epochs
# MAGIC * Declare an `MLFlowLogger` - this will be used to track our experiment parameters and metrics into MLflow. This is needed, since MLFlow `autologging` support is still being added for PyTorch Lightning > 1.5
# MAGIC * Create a `PyTorch Lightning Trainer` to train our model
# MAGIC * Train our model!

# COMMAND ----------

import mlflow
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

mlflow.autolog(disable = True)

# Explicitly setting the experiment here to the user's own folder: this is needed for the notebook to run in jobs
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/insurance_qa'.format(username))

early_stop_callback = EarlyStopping(
  monitor="val_loss",
  min_delta=0.001,
  patience=3,
  verbose=True,
  mode="min"
)

with mlflow.start_run(run_name = "distilbert") as run:

  loss = nn.CrossEntropyLoss(weight = training_data.weights)
  lit_model = LitModel(loss = loss)
  run_id = run.info.run_id
  experiment_name = mlflow.get_experiment(run.info.experiment_id)

  mlf_logger = MLFlowLogger(
      experiment_name = experiment_name,
      run_id=run_id,
      tracking_uri="databricks"
  )

  accelerator = "gpu" if torch.cuda.is_available() else "cpu"

  # Warning: max_epochs is set to 10 only for testing purposes
  # For optimal accuracy, remove the max_epochs param
  # and uncomment "callbacks = [early_stop_callback]"

  trainer = pl.Trainer(
    max_epochs = 3,
    default_root_dir = "/tmp/insuranceqa",
    logger = mlf_logger,
    accelerator = accelerator,
    log_every_n_steps = 5
    #callbacks = [early_stop_callback]
  )

# Make sure to run this on a GPU cluster, otherwise it will take longer to train
# (approx. 1.5 hours on CPU for 10 epochs)

  trainer.fit(
    lit_model,
    train_dataloaders = train_dataloader,
    val_dataloaders = test_dataloader
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Testing our model
# MAGIC 
# MAGIC Let's run inference with the model we just trained and see the results we get.

# COMMAND ----------

def predict(question = "how do i get car insurance?", ground_truth = "auto-insurance") -> str:
  """Get the specific intent from a question related to Insurance.
    question: str - The question to be asked, e.g. 'how do I get car insurance?'
    ground_truth: str - Intent that is expected for that question, e.g. 'auto-insurance'
  """

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

# DBTITLE 1,Seeing how our model performs with Test Data
for i in range(0, 10):
  # sample(1) to shuffle data
  sample = test_data.df.sample(1)
  question = sample["question_en"].values[0]
  topic = sample["topic_en"].values[0]

  pred = predict(question, topic)
  print(f"Question: {question}")
  print(f"Predicted '{topic}'? {pred == training_data.class_mappings[topic]}")
  print(f"------------------------------------------------------------------")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Initial Analysis
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC * Our model is performing poorly. This is expected - since we're setting *max_epochs = 10*, meaning our model hasn't iterated through our training set long enough to learn properly.
# MAGIC * To improve performance, try commenting *max_steps* and uncommenting *max_epochs*, and check the new results!
# MAGIC * For optimal performance, you should comment both *max_steps* and *max_epochs*, and uncomment the *callbacks* parameter in the *Trainer* declaration. Bear in mind that if training with a CPU, this will take a long time.
# MAGIC * With Early Callback and training on a GPU, you will be able to achieve **validation loss** of around 0.10 within **30 minutes**

# COMMAND ----------

# DBTITLE 1,Looking at the runs registered in MLflow for our model
import pickle

experiment = mlflow.set_experiment('/Users/{}/insurance_qa'.format(username))
runs_df = mlflow.search_runs(
  experiment_ids = [experiment.experiment_id],
  order_by = ["start_time DESC"],
  filter_string = "status = 'FINISHED'",
  output_format = "pandas"
)

target_run_id = runs_df.loc[0,"run_id"]
runs_df.head()

# COMMAND ----------

# DBTITLE 1,Registering the best performing run so far as a model
with mlflow.start_run(run_id = target_run_id, nested = True) as run:

  mlflow.pytorch.log_model(
    pytorch_model = lit_model,
    pickle_module = pickle,
    artifact_path = "model",
    pip_requirements = [
      "pytorch_lightning==1.8.6",
      "transformers==4.23.1"
    ],
    registered_model_name = "distilbert_en_uncased_insuranceqa"
  )

# COMMAND ----------


