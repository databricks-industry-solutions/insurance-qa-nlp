# Databricks notebook source
# MAGIC %md This notebook can be found at https://github.com/databricks-industry-solutions/insurance-qa-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Insurance Q&A Intent Classification with Databricks & Hugging Face
# MAGIC ### Running Inference
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rafaelvp-db/dbx-insurance-qa-hugging-face/master/img/header.png" width="800px"/>

# COMMAND ----------

import mlflow

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment = mlflow.set_experiment('/Users/{}/insurance_qa'.format(username))
runs_df = mlflow.search_runs(
  experiment_ids = [experiment.experiment_id],
  order_by = ["start_time DESC", "metrics.val_loss"],
  filter_string = "status = 'FINISHED'",
  output_format = "pandas"
)

target_run_id = runs_df.loc[0, "run_id"]
val_loss = runs_df.loc[0, "metrics.val_loss"]
print(f"Run ID '{target_run_id}' has the best val_loss metric ({val_loss}); selecting it for inference")
runs_df.head()

# COMMAND ----------

from insuranceqa.datasets.insuranceqa import InsuranceDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from insuranceqa.models.distilbert_insuranceqa import LitModel
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
import torch

from pyspark.sql.functions import struct, col

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
mlflow.artifacts.download_artifacts(
  run_id = target_run_id,
  artifact_path = "model/data",
  dst_path = "/dbfs/tmp/"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

loaded_model = torch.load(
  "/dbfs/tmp/model/data/model.pth",
  map_location = torch.device(device)
)

loaded_model = sc.broadcast(loaded_model)
tokenizer = sc.broadcast(tokenizer)

# COMMAND ----------

from torch.utils.data import DataLoader
import pytorch_lightning as pl

def predict(pdf):
  
  label_list = []
  dataset = InsuranceDataset(questions = pdf.question_en.values)
  dataloader = DataLoader(dataset, batch_size = 256, shuffle = False, num_workers = 4)
  trainer = pl.Trainer(accelerator = "gpu" if torch.cuda.is_available() else "cpu")
  pred = trainer.predict(loaded_model.value, dataloader)

  pred_list = []
  for item in pred:
    for logit in item.logits:
      pred_list.append(logit.argmax().item())

  pdf["pred"] = pred_list

  return pdf

# COMMAND ----------

test_df = pd.DataFrame(["hi I crashed my car"], columns = ["question_en"])
predict(test_df)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark import StorageLevel
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from insuranceqa.datasets.insuranceqa import InsuranceDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

spark.conf.set("spark.sql.adaptive.enabled", False)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", False)

valid_df = spark.sql("select id, question_en, topic_en from insuranceqa.valid")
valid_df = valid_df.persist(StorageLevel.MEMORY_ONLY)
valid_df.count()

predict_udf = pandas_udf(predict, returnType = IntegerType())

valid_df = valid_df.groupBy(
    F.spark_partition_id().alias("_pid")
  ).applyInPandas(
    predict,
    schema = "id string, question_en string, topic_en string, pred int"
  )

# COMMAND ----------

display(valid_df)

# COMMAND ----------

intent_df = spark.sql("select topic_id, topic_en as intent from insuranceqa.intent") \
  .withColumn("topic_id", F.col("topic_id").cast("int"))

(
  valid_df
    .join(intent_df, intent_df.topic_id == valid_df.pred)
    .write
    .saveAsTable(
      "insuranceqa.valid_pred",
      mode = "overwrite",
      mergeSchema = True
    )
)

# COMMAND ----------

display(valid_df)

# COMMAND ----------


