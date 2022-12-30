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

!pip install -q --upgrade pip && pip install -q transformers pytorch-lightning==1.8.6

# COMMAND ----------

from insuranceqa.datasets.insuranceqa import InsuranceDataset
from insuranceqa.models.distilbert_insuranceqa import LitModel
import mlflow
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType, struct, col
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_info = mlflow.models.get_model_info("models:/distilbert_en_uncased_insuranceqa/latest")
target_run_id = model_info.run_id
print(f"Model run_id: {target_run_id}")

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
  dataloader = DataLoader(dataset, batch_size = 128, shuffle = False, num_workers = 4)
  trainer = pl.Trainer(
    accelerator = "gpu" if torch.cuda.is_available() else "cpu",
    logger = False
  )
  with torch.no_grad():
    pred = trainer.predict(model = loaded_model.value, dataloaders = [dataloader], return_predictions = True)

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
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from insuranceqa.datasets.insuranceqa import InsuranceDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

spark.conf.set("spark.sql.adaptive.enabled", False)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", False)
spark.conf.set("spark.sql.shuffle.partitions", 4)

valid_df = spark.sql("select id, lemmas, question_en, topic_en from insuranceqa.valid")


# Testing purposes, comment the line below for full data
#valid_df = valid_df.sample(0.3)
#valid_df.count()

valid_df = valid_df.cache()
valid_df.count()

# Might want to increase number of partitions for higher degree of
# parallelization

valid_df = valid_df.repartition(4)
valid_df.count()

predict_udf = pandas_udf(predict, returnType = IntegerType())

valid_df = valid_df.groupBy(
    F.spark_partition_id().alias("_pid")
  ).applyInPandas(
    predict,
    schema = "id string, question_en string, lemmas string, topic_en string, pred int"
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
      "insuranceqa.valid_pred_sample",
      mode = "overwrite",
      mergeSchema = True
    )
)

# COMMAND ----------


