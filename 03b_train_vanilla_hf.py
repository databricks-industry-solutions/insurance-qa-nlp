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

!pip install -q --upgrade pip && pip install -q --upgrade transformers datasets

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

base_model = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(base_model)
model = DistilBertForSequenceClassification.from_pretrained(base_model)

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

# DBTITLE 1,Customizing our Model
# MAGIC %md
# MAGIC 
# MAGIC * We need to create a dataset in a format which is acceptable by Hugging Face
# MAGIC * We need to define how our data will be *encoded* or *tokenized*
# MAGIC * Our model must have 12 different labels; we will leverage the `AutoModelForSequenceClassification` class from Hugging Face to customise that part

# COMMAND ----------

from datasets import load_dataset

output_path = "/tmp/insurance"

def prepare_write_dataframe(df, output_path):
  output_df = (
    df
      .withColumnRenamed("question_en", "text")
      .withColumnRenamed("topic_en", "label")
      .drop("id")
  )

  output_df.write.parquet(output_path, mode = "overwrite")

train = spark.sql("select * from insuranceqa.train")
test = spark.sql("select * from insuranceqa.test")

train_path = f"{output_path}/insurance_train"
test_path = f"{output_path}/insurance_test"

prepare_write_dataframe(train, train_path)
prepare_write_dataframe(test, test_path)

dataset = load_dataset(
  "parquet", 
  data_files = {
    "train": f"/dbfs{train_path}/*.parquet",
    "test":f"/dbfs{test_path}/*.parquet"
  }
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's see how our dataset samples look like:

# COMMAND ----------

dataset["train"][123]

# COMMAND ----------

from datasets import ClassLabel, Value

labels = list(set(dataset["train"]["label"]))
new_features = dataset["train"].features.copy()
new_features["label"] = ClassLabel(names = labels)
dataset = dataset.cast(new_features)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Data Tokenization

# COMMAND ----------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding = True, truncation = True, return_tensors = "pt")

dataset_tokenized = dataset.map(tokenize_function, batched=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Training our Model
# MAGIC 
# MAGIC Here we will:
# MAGIC 
# MAGIC * Create a `Trainer` object - this is a helper class from Hugging Face which makes training easier
# MAGIC * Instantiate a `TrainingArguments` object
# MAGIC * Create an EarlyStoppingCallback - this will help us avoid our model overfit
# MAGIC * Train our model

# COMMAND ----------

label2id = dataset["train"].features["label"]._str2int
id2label = dataset["train"].features["label"]._int2str

model = AutoModelForSequenceClassification.from_pretrained(
  base_model,
  num_labels = len(labels),
  label2id = label2id,
  id2label = id2label
)

# COMMAND ----------

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import mlflow

mlflow.end_run()

# Here we define a DataCollator which combines different samples in the dataset and pads them to the same length
data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
  output_dir = "insurance_qa",
  evaluation_strategy = "epoch",
  num_train_epochs = 1
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset_tokenized["train"],
    eval_dataset = dataset_tokenized["test"],
    data_collator = data_collator
)

trainer.train()

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


