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

# DBTITLE 1,Getting the model's latest version
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_info = client.get_latest_versions(name = "insuranceqa")[0]
model_info

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


