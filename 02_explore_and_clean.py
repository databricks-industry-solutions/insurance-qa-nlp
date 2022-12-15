# Databricks notebook source
# MAGIC %md This notebook can be found at https://github.com/databricks-industry-solutions/insurance-qa-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Insurance Q&A Intent Classification with Databricks & Hugging Face
# MAGIC ### Data Exploration
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rafaelvp-db/dbx-insurance-qa-hugging-face/master/img/header.png" width="800px"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Looking at the amount of intents

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from insuranceqa.train

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Distribution of question lengths

# COMMAND ----------

from pyspark.sql.functions import col, length, median

df_summary = (
  spark.sql("select * from insuranceqa.train")
    .withColumn("length", length(col("question_en")))
    .select("length")
    .summary()
)

display(df_summary)

# COMMAND ----------

# DBTITLE 1,Initial Insights
# MAGIC %md
# MAGIC 
# MAGIC * There's a really long tail in terms of questions lengths
# MAGIC * As a rule of thumb, questions that are this long are usually not that useful / insightful; we'll consider these as anomalies and drop them
# MAGIC * This will be useful when it comes to tokenizing these inputs, as we'll save some valuable GPU memory that would otherwise be wasted with zeros due to padding (truncating these samples could make the model confused)

# COMMAND ----------

def remove_outliers(df):

  df = (
    df
      .withColumn("length", length(col("question_en")))
      .filter("length < 50") # Limit size to approx. 75th quantile
      .drop("length")
  )
  return df

# COMMAND ----------

for table in ["insuranceqa.train", "insuranceqa.test", "insuranceqa.valid"]:

  df = spark.sql(f"select * from {table}")
  df = remove_outliers(df)
  df.write.saveAsTable(name = table, mode = "overwrite")

# COMMAND ----------


