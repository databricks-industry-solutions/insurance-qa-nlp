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

valid_df = spark.sql("select * from insuranceqa.valid")
display(valid_df)

# COMMAND ----------

from insuranceqa.datasets.insuranceqa import InsuranceDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from insuranceqa.models.distilbert_insuranceqa import LitModel
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
import torch
import mlflow
from pyspark.sql.functions import struct, col

logged_model = 'runs:/a78e2612f80f47d0a70be0c4a86b09b1/model'
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
mlflow.artifacts.download_artifacts(
  run_id = "a78e2612f80f47d0a70be0c4a86b09b1",
  artifact_path = "model/data",
  dst_path = "/dbfs/tmp/"
)
loaded_model = torch.load("/dbfs/tmp/model/data/model.pth", map_location = torch.device("cpu"))
loaded_model = sc.broadcast(loaded_model)
tokenizer = sc.broadcast(tokenizer)

# COMMAND ----------

def predict(pdf):
  
  label_list = []
  for question in pdf.question_en.values:
    encodings = tokenizer.value(
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
      y_hat = loaded_model.value(**inputs)
      label = y_hat.logits.argmax().item()
      label_list.append(label)

  pdf["pred"] = label_list
  return pdf

# COMMAND ----------

test_df = pd.DataFrame(["hi I crashed my car"], columns = ["question_en"])
predict(test_df)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark import StorageLevel
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark.conf.set("spark.sql.adaptive.enabled", False)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", False)
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 50)

valid_df = spark.sql("select question_en from insuranceqa.valid")
valid_df = valid_df.persist(StorageLevel.MEMORY_ONLY)
valid_df.count()
valid_df = valid_df.repartition(32)
valid_df.count()

predict_udf = pandas_udf(predict, returnType = IntegerType())
valid_df = valid_df.groupBy(F.spark_partition_id().alias("_pid")).applyInPandas(
  predict,
  schema = "question_en string, pred int"
)

# COMMAND ----------

valid_df.write.saveAsTable("insuranceqa.valid_pred", mode = "overwrite", mergeSchema = True)
