# Databricks notebook source
# MAGIC %md This notebook can be found at https://github.com/databricks-industry-solutions/insurance-qa-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Insurance Q&A Intent Classification with Databricks & Hugging Face
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rafaelvp-db/dbx-insurance-qa-hugging-face/master/img/header.png" width="800px"/>

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
# MAGIC * For the purpose of this accelerator, we are leveraging the [Insurance QA Dataset](https://github.com/shuzi/insuranceQA)
# MAGIC * In this notebook, we will **download** and **ingest** this dataset into multiple Delta tables for **train**, **test** and **validation** sets
# MAGIC * In the next notebook (*02_explore_clean.py*), we will do a bit of data exploration and cleaning
# MAGIC * Next, we will fine tune an NLP model (*distilbert-en*) with our data
# MAGIC * We will create a Pandas UDF to wrap the model that we fine tuned. This will make it possible for us to generate predictions on top of both **static/batch** data sources and **streaming** ones.
# MAGIC * Finally, we will deploy our model as a **realtime** prediction endpoint by leveraging [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)

# COMMAND ----------

# MAGIC %pip install datasets

# COMMAND ----------

# MAGIC %run ./config/notebook-config

# COMMAND ----------

# DBTITLE 1,Downloading the Insurance QA Dataset
from datasets import load_dataset

dataset = load_dataset("j0selit0/insurance-qa-en")
dataset

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## First Glance at the Dataset
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC * Now that that dataset was downloaded, let's have a look at its contents.
# MAGIC * From the previous cell, we can see that there are two main columns of interest: *question_en* and *topic_en*. The first one contains questions related to multiple insurance topics, while the second stores the classification/topic for each question.
# MAGIC * Let's go on and have a glance at same data samples from the training set. Hugging Face Datasets have a custom format (*DatasetDictionary*), but luckily there are some quite handy functions, such as *to_pandas()*, which allows us to explore the data contents in a more intuitive way.

# COMMAND ----------

display(dataset["train"].to_pandas().loc[:10, ["question_en", "topic_en"]])

# COMMAND ----------

# DBTITLE 1,Basic Cleaning
# Let's convert everything to lower case and remove extra spaces

import re
from datasets import ClassLabel

def clean(example: str) -> str:

  output = []
  for question in example["question_en"]:
    question_clean = question.lower()
    question_clean = re.sub(' {2,}', ' ', question_clean)
    output.append(question_clean)
  
  example["question_en"] = output
  return example

clean_dataset = dataset.map(lambda example: clean(example), batched = True)

# Renaming our column and converting labels to ClassLabel

clean_dataset = clean_dataset.remove_columns(["index"])
clean_dataset = clean_dataset.rename_columns({"question_en": "text", "topic_en": "label"})
names = list(set(clean_dataset["train"]["label"]))
clean_dataset = clean_dataset.cast_column("label", ClassLabel(names = names))

# Save to cleaned dataset for further training
## We first save the clean dataset to a local path in the driver, then copy it to DBFS because pyarrow is unable to write directly to DBFS
local_path = "/tmp/insuranceqa"
dbutils.fs.rm(config["main_path"], True)
clean_dataset.save_to_disk(local_path)
dbutils.fs.cp(f"file:///{local_path}", config["main_path"], recurse = True)

# COMMAND ----------

# DBTITLE 1,Generating a data profile
# By using the display function, we can easily generate a data profile for our dataset

display(dataset["train"].to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * Looking at the profile above, we can notice that questions related to *life insurance* are quite frequent. As an insurance company, we might be interested into taking actions to leverage this aspect within our audience - for instance, marketing, sales or educational campaigns. On the other hand, it could also be that our customer service team needs to be enabled or scaled. We will look at the wider distribution of topics/intents in order to look for more insights.
# MAGIC 
# MAGIC * To achieve that, run the cell below, and simply click on the plus sign next to the "Table" tab. You can then create a Bar Plot visualization, where the X-axis contains the *topic_en* column, and the Y axis contains a *COUNT* of the *index* column.

# COMMAND ----------

display(dataset["train"].to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * Looking at the distribution of topics/intents across our training set, we can see that apart from life insurance, auto insurance and medicare are quite popular themes.
# MAGIC * When we look at the opposite direction - less popular intents, we can highlight *critical illness insurance*, *long term care insurance* and *other insurance*. Here, we might also be interested in understanding more about these specific liness of businesses, and even compare profit margins across them. The fact that there are few questions around these topics could also mean that we are doing a better job at enabling customers to solve their problems or answer their questions through digital channels, without having to talk to a human agent.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Here we create the table by converting a HuggingFace dataset to Pandas, then to write as a Delta table. This may have scalability contraints as the dataset cannot be larger than your driver memory.
# MAGIC 
# MAGIC In more realistic scenarios, your data may already come in formats that Spark can process in parallelized ways. 

# COMMAND ----------

# DBTITLE 1,Saving the test set into Delta for inference
test_df = spark.createDataFrame(dataset["test"].to_pandas())
test_df.write.saveAsTable("questions", mode = "overwrite")

# COMMAND ----------

# DBTITLE 1,Next Steps
# MAGIC %md
# MAGIC 
# MAGIC * Now that we have a dataset and we have done enough exploration, we will proceed to training our model.
# MAGIC * Our model will be used to classify customer questions across 12 different topics.
# MAGIC * To get started with training, you can refer to the <a href='#notebook/3012450039806056/' target='_blank'>02_train</a> notebook in this repo.
