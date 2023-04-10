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
import torch

client = MlflowClient()
model_info = [model for model in client.get_latest_versions(name = "insuranceqa") if model.current_stage not in ["Archived", "Production"]][0]
pipeline = mlflow.pyfunc.load_model(f"models:/{model_info.name}/{model_info.version}")
model_info

# COMMAND ----------

# DBTITLE 1,Defining a prediction function and running a simple test
import pandas as pd
from typing import List

def predict(questions: pd.Series) -> pd.Series:
  """Wrapper function for the pipeline we created in the previous step."""

  result = pipeline.predict(questions.to_list())
  return pd.Series(result)

# Simple test
simple_df = pd.DataFrame(["hi I crashed my car"], columns = ["question_en"])
test_prediction = predict(simple_df.question_en)
test_prediction.values

# COMMAND ----------

# DBTITLE 1,Promoting the model to production
if test_prediction is not None:
  # Prediction is valid, so we can transition our model version to Production stage
  client.transition_model_version_stage(
    name = model_info.name,
    version = model_info.version,
    stage = "Production",
    archive_existing_versions = True
  )

# COMMAND ----------

# DBTITLE 1,Running inference on top of a Delta Table
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

spark.conf.set("spark.sql.adaptive.enabled", False)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", False)
spark.conf.set("spark.sql.shuffle.partitions", 6)

test_df = spark.sql("select question_en, topic_en from insuranceqa.questions")

# Testing purposes, comment the line below for full data
test_df = test_df.sample(0.3)
test_df.count()

test_df = test_df.cache()
test_df.count()

# Might want to increase number of partitions for higher degree of
# parallelization

test_df = test_df.repartition(4)
test_df.count()

predict_udf = F.pandas_udf(predict, returnType = StringType())

test_df = (
  test_df
    .withColumn("predicted", predict_udf(F.col("question_en")))
)

# COMMAND ----------

display(test_df)

# COMMAND ----------

# DBTITLE 1,Save prediction results into Delta
(
  test_df
    .write
    .saveAsTable(
      "insuranceqa.predictions",
      mode = "overwrite",
      mergeSchema = True
    )
)

# COMMAND ----------

# DBTITLE 1,Calculating Performance Metrics
# We calculate the amount of correct predictions and divide by the total number of predictions

pred_df = spark.sql("select topic_en, predicted from insuranceqa.predictions")
num_correct_predictions = pred_df.filter("topic_en = predicted").count()
accuracy = num_correct_predictions / pred_df.count()

print(f"Accuracy score: {accuracy}")

# COMMAND ----------


