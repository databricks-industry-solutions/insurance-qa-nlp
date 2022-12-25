# Databricks notebook source
# MAGIC %md This notebook can be found at https://github.com/databricks-industry-solutions/insurance-qa-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Insurance Q&A Intent Classification with Databricks & Hugging Face
# MAGIC ### Data Ingestion
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rafaelvp-db/dbx-insurance-qa-hugging-face/master/img/header.png" width="800px"/>

# COMMAND ----------

!rm -rf /dbfs/tmp/word2vec-get-started && rm -rf /tmp/word2vec-get-started/
%cd /tmp
!git clone https://github.com/rafaelvp-db/word2vec-get-started
!mv /tmp/word2vec-get-started/corpus /dbfs/tmp/word2vec-get-started

# COMMAND ----------

dbfs_path = "/dbfs/tmp/word2vec-get-started/insuranceqa/questions"
!ls {dbfs_path}/*.txt

# COMMAND ----------

from pyspark.sql.functions import (
  lower,
  regexp_replace,
  col,
  monotonically_increasing_id
)

def ingest_data(
  path,
  output_table,
  database = "insuranceqa"
):

  spark.sql(f"create database if not exists {database}")
  spark.sql(f"drop table if exists {output_table}")

  df = spark.read.csv(
    path,
    sep = "\t",
    header = True
  )

  # Read CSV Q&A data into dataframe
  df = df.toDF(
    'id',
    'topic_en',
    'topic_jp',
    'question_en',
    'question_jp'
  )\
  .select("id", "topic_en", "question_en")

  return df

def clean(df):

  df = df.withColumn(
    "question_en",
    regexp_replace(lower(col("question_en")), "  ", " ")
  )
  return df

def pipeline(path, output_table, database = "insuranceqa"):
  df = ingest_data(path, output_table)
  df = clean(df)
  df_intents = (
    df
      .select("topic_en")
      .distinct()
      .orderBy("topic_en")
      .withColumn("topic_id", monotonically_increasing_id())
  )
  df_intents.write.saveAsTable(
    f"{database}.intent",
    mode = "overwrite",
    mergeSchema = True
  )
  df.write.saveAsTable(output_table)

splits = ["train", "test", "valid"]
for split in splits:
  pipeline(
    f"{dbfs_path.replace('/dbfs', '')}/{split}.questions.txt",
    f"insuranceqa.{split}"
  )

# COMMAND ----------

for table in ["train", "test", "valid", "intent"]:
  count = spark.sql(f"select count(1) from insuranceqa.{table}").take(1)
  print(f"Table '{table}' has {count} rows")
