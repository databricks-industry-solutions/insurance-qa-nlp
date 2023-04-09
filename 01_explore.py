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

clean_dataset.save_to_disk("/tmp/insurance")

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
# MAGIC ## [Optional] A Deeper Dive Into Customer Questions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * We are now able to get an idea about how question intents are distributed across our training set, which has been previously labelled by specialized professionals. What about when we are looking at data which hasn't been labelled yet?
# MAGIC * In this case, we need to resort to an unsupervised, no-labels approach.
# MAGIC * For this exercise, we will assume that we still don't have the labels for questions which are part of the dataset. We will then analyse the raw text for questions from the testing set, and by using BERT Embeddings, we will try to have an idea about which topics are more common in our testing set.
# MAGIC * **BERTopic** is a Python library which leverages BERT embeddings for performing unsupervised topic modelling. By using it, we can group our text data into different clusters, and also visually inspect them.

# COMMAND ----------

!pip install -q bertopic

# COMMAND ----------

from bertopic import BERTopic

docs = dataset["test"].to_pandas().question_en.values
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

# COMMAND ----------

# Visualizing topics

topic_model.get_topic_info().head(10)

# COMMAND ----------

# Looking at how many topics

topic_model.get_topic_info().Name.unique()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Interesting to see that according to our model which uses BERT Embeddings, we have 32 different topics in our testing set. This is higher than the amount of classifications that we have in our training set - potentially, our list of unsupervised topics is a bit more fine grained than the topics that we are using in our training set.
# MAGIC 
# MAGIC Nevertheless, would be great to have a closer look at how these topics are distributed, and some examples of questions which are part of each of them. The great news is that BERTopic has a function to easily achieve that:

# COMMAND ----------

topic_model.visualize_topics()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * The visualization above shows us an *Intertopic Distance Map*, a visualization of the topics in a two-dimensional space . The radius of the topic circles is proportional to the amount of words that belong to each topic in our document.
# MAGIC * Ideally, intertopic distance across similar topics should be low, and the opposite should be true for topics which have differ in meaning or semantics.
# MAGIC * While the visualization above is helpful, visually inspecting examples of our questions and how they relate to each other in terms of topic clusters will help us understand our data even more. In order to do that, we will leverage two additional libraries: [Sentence Transformers](https://www.sbert.net/) and [UMap](https://umap-learn.readthedocs.io/en/latest/).

# COMMAND ----------

!pip install -q umap-learn sentence-transformers

# COMMAND ----------

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP

# Prepare embeddings
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=False)

# Train BERTopic
topic_model = BERTopic().fit(docs, embeddings)

# Run the visualization with the original embeddings
topic_model.visualize_documents(docs, embeddings=embeddings)

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)

# COMMAND ----------

# DBTITLE 1,Topic Modeling Takeaways
# MAGIC %md
# MAGIC 
# MAGIC * The visualization above is interactive, which allows us to zoom in different topic clusters.
# MAGIC * For instance, if we look at the cluster at the center of this plot, we can notice that there multiple occurrencies of topics related to life insurance.
# MAGIC * We can also hover across different points across the same color/cluster; that will give us the chance to look at sample data points that were included in that particular cluster.

# COMMAND ----------

# DBTITLE 1,Next Steps
# MAGIC %md
# MAGIC 
# MAGIC * Now that we have a dataset and we have done enough exploration, we will proceed to training our model.
# MAGIC * Our model will be used to classify customer questions across 12 different topics.
# MAGIC * To get started with training, you can refer to the <a href='#notebook/3012450039806056/' target='_blank'>02_train</a> notebook in this repo.
