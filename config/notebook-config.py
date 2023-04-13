# Databricks notebook source
config = {}
config["model_name"] = "insuranceqa"
config["database_name"] = "insuranceqa"
config["model_output_dir"] = "/tmp/insuranceqa_model"
config["pipeline_output_dir"] = "/tmp/insuranceqa_pipeline/artifacts"
config["model_artifact_path"] = "model"
config["main_path"] = "dbfs:/tmp/insurance"
config["main_path_w_dbfs"] = "/dbfs/tmp/insurance"
config["main_local_path"] = "file:/tmp/insurance"

# COMMAND ----------

spark.sql(f"""create database if not exists {config["database_name"]}""")
spark.sql(f"""use {config["database_name"]}""")

# COMMAND ----------

import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/insuranceqa'.format(username))
