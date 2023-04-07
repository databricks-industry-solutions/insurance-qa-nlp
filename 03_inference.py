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

# DBTITLE 1,Getting best performing experiment runs
import mlflow

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment = mlflow.get_experiment_by_name(f"/Repos/{user_name}/insurance-qa-nlp/02_train")

runs_df = mlflow.search_runs(
  experiment_ids = [experiment.experiment_id],
  order_by = ["metrics.eval_loss"],
  filter_string = "status = 'FINISHED'",
)

target_run_id = runs_df.loc[0, "run_id"]
runs_df.head()

# COMMAND ----------

# DBTITLE 1,Registering the best run as a model
model_name = "insuranceqa"
base_model = "distilbert-base-uncased"

model_info = mlflow.register_model(
  model_uri = f"runs:/{target_run_id}/model",
  name = "insuranceqa",
  tags = {"base_model": base_model}
)

# COMMAND ----------

!ls {dst_path}/model

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

dst_path = "/tmp/insuranceqa/"
model_path = mlflow.artifacts.download_artifacts(f"runs:/{model_info.run_id}/model", dst_path = dst_path)

tokenizer = AutoTokenizer.from_pretrained(base_model)
config = AutoConfig.from_pretrained(base_model)
model = AutoModelForSequenceClassification.from_pretrained(f"{model_path}/data", config = config)

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


