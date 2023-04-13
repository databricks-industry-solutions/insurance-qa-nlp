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

# MAGIC %pip install -q datasets

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config/notebook-config

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Distilbert Example
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC * In the cell below, we simply instantiate a Hugging Face Text Classification Pipeline to see how it works.
# MAGIC * We go on and run a sample prediction for a piece of text

# COMMAND ----------

from transformers import pipeline

pipe = pipeline("text-classification")
pipe(["This restaurant is awesome", "This restaurant is awful"])

# COMMAND ----------

# DBTITLE 1,Fine Tuning our Model
# MAGIC %md
# MAGIC 
# MAGIC * We need to create a dataset in a format which is acceptable by Hugging Face
# MAGIC * We need to define how our data will be *encoded* or *tokenized*
# MAGIC * Our model must have 12 different labels; we will leverage the `AutoModelForSequenceClassification` class from Hugging Face to customise that part

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Reading the dataset

# COMMAND ----------

import datasets

local_path = "/tmp/insurance"
dbutils.fs.rm(f"file://{local_path}", recurse = True)
dbutils.fs.cp(config["main_path"], f"file://{local_path}", recurse = True)
dataset = datasets.load_from_disk(local_path)

# COMMAND ----------

# DBTITLE 1,Dataset Tokenization
from transformers import AutoTokenizer

base_model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model)

def tokenize(examples):
  return tokenizer(examples["text"], padding = True, truncation = True, return_tensors = "pt")

tokenized_dataset = dataset.map(tokenize, batched = True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Training our Model
# MAGIC 
# MAGIC Here we will:
# MAGIC 
# MAGIC * Create a `Trainer` object - this is a helper class from Hugging Face which makes training easier
# MAGIC * Instantiate a `TrainingArguments` object
# MAGIC * Create an `EarlyStoppingCallback` - this will help us avoid our model overfit
# MAGIC * Train our model

# COMMAND ----------

from transformers import AutoModelForSequenceClassification
import datasets

dataset = datasets.load_from_disk("/dbfs/tmp/insurance")
label2id = dataset["train"].features["label"]._str2int
id2label = dataset["train"].features["label"]._int2str

model = AutoModelForSequenceClassification.from_pretrained(
  base_model,
  num_labels = len(label2id),
  label2id = label2id,
  id2label = dict(enumerate(id2label))
)

# COMMAND ----------

# DBTITLE 1,Quick Glance at DistilBERT's Architecture
model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Defining Class Weights
# MAGIC 
# MAGIC </br>
# MAGIC 
# MAGIC * In the previous notebook, we could observe the distribution of intents is varied, with a prevalence of questions around `life-insurance`.
# MAGIC * This can be a challenge for our model to be able to generalize.
# MAGIC * We can define different class weights, so that wrong predictions for a particular intent or topic are *penalized* differently depending on how frequent they are in our training set.
# MAGIC * To achieve this, we will calculate *class weights* for the 12 intents in our dataset - the least frequent an intent is, the higher the penalty it will generate in case of wrong predictions.

# COMMAND ----------

from collections import Counter

def get_class_weights(labels):

  counter = Counter(labels)
  item_count_dict = dict(counter.items())
  size = len(labels)
  weights = list({k: (size / v) for k, v in sorted(item_count_dict.items())}.values())
  return weights

weights = get_class_weights(dataset["train"]["label"])

# COMMAND ----------

import mlflow
import itertools
import torch

from transformers import (
  TrainingArguments,
  Trainer,
  DataCollatorWithPadding,
  EarlyStoppingCallback,
  ProgressCallback
)

# Here we define a DataCollator which combines different samples in the dataset and pads them to the same length
data_collator = DataCollatorWithPadding(tokenizer)

# We use an EarlyStoppingCallback so that trained is stopped when performance metrics are no longer improving
early_stopping = EarlyStoppingCallback(
  early_stopping_patience = 3,
  early_stopping_threshold = 0.01
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(torch.device(device))

training_args = TrainingArguments(
  output_dir = "/tmp/insurance_qa",
  evaluation_strategy = "steps",
  eval_steps = 100,
  num_train_epochs = 1, # Increase for better accuracy
  per_device_train_batch_size = 128,
  per_device_eval_batch_size = 128,
  load_best_model_at_end = True,
  learning_rate = 2e-5,
  weight_decay = 0.01,
  xpu_backend = "ccl",
  no_cuda = False if torch.cuda.is_available() else True
)

class CustomTrainer(Trainer):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    # forward pass
    outputs = model(**inputs)
    logits = outputs.get('logits')
    # compute custom loss
    device = 0 if torch.cuda.is_available() else -1
    weights_tensor = torch.tensor(weights).to(device)
    loss_fct = torch.nn.CrossEntropyLoss(weight = weights_tensor)
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["valid"],
    data_collator = data_collator,
    callbacks = [early_stopping, ProgressCallback()]
)

# COMMAND ----------

# DBTITLE 1,Train Model and Save Artifacts
from transformers import pipeline
import logging

class InsuranceQAModel(mlflow.pyfunc.PythonModel):

  def load_context(self, context):
    device = 0 if torch.cuda.is_available() else -1
    pipeline_path = context.artifacts["insuranceqa_pipeline"]
    model_path = context.artifacts["base_model"]
    self.pipeline = pipeline(
      "text-classification",
      model = context.artifacts["insuranceqa_pipeline"],
      config = context.artifacts["base_model"],
      device = device
    )
    
  def predict(self, context, model_input):
    try:
      logging.info(f"Model input: {model_input}")
      questions = list(model_input)

      results = self.pipeline(questions, truncation = True, batch_size = 8)
      labels = [result["label"] for result in results]
      logging.info(f"Model output: {labels}")
      return labels

    except Exception as exception:
      logging.error(f"Model input: {questions}, type: {str(type(questions))}")
      return {"error": str(exception)}

# COMMAND ----------

from transformers import pipeline
import transformers
import numpy as np

model_output_dir = "/tmp/insuranceqa_model"
pipeline_output_dir = "/tmp/insuranceqa_pipeline/artifacts"
model_artifact_path = "model"

mlflow.set_experiment(experiment_name = "/Shared/insuranceqa_distilbert")

with mlflow.start_run() as run:
  trainer.train()
  trainer.save_model(model_output_dir)
  pipe = pipeline(
    "text-classification",
    model = model,
    config = model.config,
    batch_size = 8,
    tokenizer = tokenizer
  )
  pipe.save_pretrained(pipeline_output_dir)

  # Log custom PyFunc model
  mlflow.pyfunc.log_model(
    artifacts = {
      "insuranceqa_pipeline": pipeline_output_dir,
      "base_model": model_output_dir
    },
    artifact_path = model_artifact_path,
    python_model = InsuranceQAModel(),
    pip_requirements = [
      f"""torch=={torch.__version__.split("+")[0]}""",
      f"""transformers=={transformers.__version__}"""
    ]
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Initial Analysis
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC * We trained our text classification model for only one epoch, so we should expect it to have a suboptimal performance.
# MAGIC * As a follow up action, we could increase the number of epochs - that would give the model more insights on different parts of our dataset, as well as different possible intents, thus increasing its ability to generalize.

# COMMAND ----------

# DBTITLE 1,Fetching the best performing run and registering a model
runs = mlflow.search_runs(
  order_by = ["start_time DESC"],
  filter_string = "attributes.status = 'FINISHED'"
)

target_run_id = runs.loc[0, "run_id"]
logged_model_uri = f"runs:/{target_run_id}/model"
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)
loaded_model

# COMMAND ----------

# DBTITLE 1,Run test prediction
loaded_model.predict(["my car broke, what should I do?"])

# COMMAND ----------

# DBTITLE 1,Registering the model
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import List
client = MlflowClient()

# Register the model
model_details = mlflow.register_model(logged_model_uri, config["model_name"])

# Simple test

model_uri = f"""models:/{config["model_name"]}/latest"""
pipeline = mlflow.pyfunc.load_model(model_uri = model_uri)

simple_df = pd.DataFrame(["hi I crashed my car"], columns = ["question_en"])
test_prediction = pipeline.predict(simple_df.question_en)
print(f"Prediction: {test_prediction}")

if test_prediction is not None:

  # Transition the model to "Production" stage in the registry
  client.transition_model_version_stage(
    name = config["model_name"],
    version = model_details.version,
    stage="Production",
    archive_existing_versions=True
  )
