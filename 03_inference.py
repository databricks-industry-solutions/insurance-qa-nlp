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

# MAGIC %pip install datasets

# COMMAND ----------

# MAGIC %run ./config/notebook-config

# COMMAND ----------

# DBTITLE 1,Reading in questions for inference
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

# ??????? explain why this is set
spark.conf.set("spark.sql.adaptive.enabled", False)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", False)

# Read questions
test_df = spark.sql("select question_en, topic_en from questions")

# Increase parallelism to at least the number of worker cores in the cluster - if the data volume is larger, you can set this to multiples of the number of worker cores
sc = spark._jsc.sc() 
worker_count = len([executor.host() for executor in sc.statusTracker().getExecutorInfos() ]) -1
total_worker_cores = spark.sparkContext.defaultParallelism * worker_count
test_df = test_df.repartition(total_worker_cores)

# COMMAND ----------

# DBTITLE 1,Running inference on top of a Delta Table
# Get model udf
import pandas as pd
from typing import List

# Loading our production model and wrap it in a UDF
prod_model_uri = f"""models:/{config["model_name"]}/production"""
pipeline = mlflow.pyfunc.load_model(prod_model_uri)

def predict(questions: pd.Series) -> pd.Series:
  """Wrapper function for the pipeline we created in the previous step."""

  result = pipeline.predict(questions.to_list())
  return pd.Series(result)

predict_udf = F.pandas_udf(predict, returnType = StringType())

# Perform inference with the UDF
test_df = (
  test_df
    .withColumn("predicted", predict_udf(F.col("question_en")))
)

# COMMAND ----------

# DBTITLE 1,Save prediction results into Delta
(
  test_df
    .write
    .saveAsTable(
      "predictions",
      mode = "overwrite",
      mergeSchema = True
    )
)

predictions = spark.sql("select * from predictions")
display(predictions)

# COMMAND ----------

# DBTITLE 1,Calculating Performance Metrics
# We calculate the amount of correct predictions and divide by the total number of predictions

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'

pred_df = predictions.toPandas()
pred_df["hit"] = pd.to_numeric(pred_df["topic_en"] == pred_df["predicted"])
accuracy_per_intent = pred_df.groupby("topic_en").hit.mean().reset_index()

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4))
plt.xticks(rotation = 45)
sns.barplot(
  y = accuracy_per_intent["topic_en"],
  x = accuracy_per_intent["hit"],
  palette = "mako",
  ax = ax
)
plt.title("Prediction Accuracy per Intent")
