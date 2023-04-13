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

# MAGIC %pip install -q datasets

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup configs
# MAGIC %run ./config/notebook-config

# COMMAND ----------

# DBTITLE 1,Reading in questions for inference
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

# Read questions
test_df = spark.sql("select question_en, topic_en from questions")

# Increase parallelism to at least the number of worker cores in the cluster - if the data volume is larger, you can set this to multiples of the number of worker cores
sc = spark._jsc.sc() 
worker_count = max(1, len([executor.host() for executor in sc.statusTracker().getExecutorInfos() ]) - 1) 
total_worker_cores = spark.sparkContext.defaultParallelism * worker_count
test_df = test_df.repartition(total_worker_cores)

# COMMAND ----------

# DBTITLE 1,Loading the model and generating test predictions
from typing import List
import pandas as pd
from pyspark.sql import functions as F
from mlflow.tracking import MlflowClient
import mlflow

# Loading our model and wrapping it in a UDF
pipeline = mlflow.pyfunc.load_model(f"models:/{config['model_name']}/production")

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

# DBTITLE 1,Saving the predictions to Delta
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
