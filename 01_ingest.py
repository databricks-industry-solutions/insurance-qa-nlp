# Databricks notebook source
# MAGIC %md This notebook can be found at https://github.com/databricks-industry-solutions/insurance-qa-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Insurance Q&A Intent Classification with Databricks & Hugging Face
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rafaelvp-db/dbx-insurance-qa-hugging-face/master/img/header.png" width="100%"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC While companies across industries have accelerated digital adoption after the COVID-19 pandemic, are insurers meeting the ever-changing demands from customers?
# MAGIC 
# MAGIC As an insurer, are you spending most of your time and resources into **creating business value**?
# MAGIC 
# MAGIC **Customer service** is a vital part of the insurance business. This is true for multiple business cases: from **marketing**, to **customer retention**, and **claims**. 
# MAGIC 
# MAGIC At the same time, turnover in customer service teams is significantly higher than others, while training such teams takes time and effort. What's more, the fact that insurance companies frequently outsource customer service to different players also represents a challenge in terms of **service quality** and **consistency**.
# MAGIC 
# MAGIC By **digitalizing** these processes, insurers can seamlessly:
# MAGIC 
# MAGIC * **Increase customer satisfaction** by **reducing waiting times**
# MAGIC * Provide a better, **interactive experience** by reducing amount of phone calls
# MAGIC * Reduce their phone bill costs
# MAGIC * **Scale** their operations by being able to do more with less
# MAGIC * Shift **money** and **human resources** from **operational** processes to actual **product** and **value creation**.
# MAGIC 
# MAGIC This solutions accelerator is a head start on developing and deploying a **machine learning solution** to detect customer intents based on pieces of unstructured text from an **Interactive Voice Response (IVR)** stream, or from a **virtual agent** - which could be integrated with a mobile app, SMS, Whatsapp and communication channels.
# MAGIC 
# MAGIC ## Target Solution
# MAGIC 
# MAGIC <img src="https://github.com/rafaelvp-db/dbx-insurance-qa-hugging-face/blob/master/img/Insurance(1).png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Getting Started
# MAGIC 
# MAGIC <hr/>
# MAGIC 
# MAGIC * For the purpose of this accelereator, we are leveraging the [Insurance QA Dataset](https://github.com/shuzi/insuranceQA)
# MAGIC * In this notebook, we will **download** and **ingest** this dataset into multiple Delta tables for **train**, **test** and **validation** sets
# MAGIC * In the next notebook (*02_explore_clean.py*), we will do a bit of data exploration and cleaning
# MAGIC * Next, we will fine tune an NLP model (*distilbert-en*) with our data
# MAGIC * Finally, we will create a Pandas UDF to wrap the model that we fine tuned. This will make it possible for us to generate predictions on top of both **static/batch** data sources and **real time / streaming** ones.

# COMMAND ----------

# DBTITLE 1,Downloading the Insurance QA Dataset
input_path = "/tmp/word2vec-get-started/"
dbfs_path = f"/dbfs{input_path}"
full_path = f"{dbfs_path}insuranceqa/questions"

!rm -rf {dbfs_path} && rm -rf {input_path}
%cd /tmp
!git clone https://github.com/rafaelvp-db/word2vec-get-started
!mv {input_path}corpus {dbfs_path}
!ls {full_path}/*.txt

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Ingesting and Cleaning our Data
# MAGIC 
# MAGIC <hr/>
# MAGIC 
# MAGIC * In the cells below, we remove some unnecessary columns, and also do some basic cleaning: converting text to lower case and removing invalid characeters
# MAGIC * We then proceed to create a Delta Table for each of the files in our dataset (train, test and valid)
# MAGIC * We also create a mapping table for the different question intents we have in the dataset. This will be useful further on when we start training our model.

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
  spark.sql(f"drop table if exists insuranceqa.{split}")
  pipeline(
    f"{full_path.replace('/dbfs', '')}/{split}.questions.txt",
    f"insuranceqa.{split}"
  )

# COMMAND ----------

# DBTITLE 1,Checking the size of our tables
for table in ["train", "test", "valid", "intent"]:
  count = spark.sql(f"select count(1) from insuranceqa.{table}").take(1)
  print(f"Table '{table}' has {count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Great, we successfully ingested our data into Delta! Now it's time to move to the next notebook to do some exploration and cleaning.
