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
# MAGIC <img src="https://raw.githubusercontent.com/rafaelvp-db/dbx-insurance-qa-hugging-face/master/img/header.png" width="100%"/>

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Looking the different intents in our dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from insuranceqa.intent

# COMMAND ----------

# DBTITLE 1,Distribution of Intents
# MAGIC %sql
# MAGIC 
# MAGIC select * from insuranceqa.train

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Distribution of question lengths

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.sql("select * from insuranceqa.train")

df_summary = (
  df
    .withColumn("sentence_len", F.length(F.col("question_en")))
    .select("sentence_len")
    .summary()
)

display(df_summary)

# COMMAND ----------

# DBTITLE 1,Initial Insights
# MAGIC %md
# MAGIC 
# MAGIC * Question intents are unevenly distributed; most questions are about **life insurance**, while very few of them relate to **critical illness insurance**
# MAGIC * There's a really **long tail** in terms of questions lengths; the majority of questions has **42 characters or less** - while a minority of them has **50 characters or more**, with max length reaching **277 characters**
# MAGIC * As a rule of thumb, questions that are this long are usually not that **useful / insightful**; we'll consider these as anomalies and drop them from our training set
# MAGIC * This will be useful when it comes to **encoding / tokenizing** these inputs. We'll save some valuable **GPU memory** that would otherwise be wasted with zeros due to padding - while truncating these samples could make the model confused

# COMMAND ----------

def remove_outliers(df):

  df = (
    df
      .withColumn("sentence_len", F.length(F.col("question_en")))
      .filter("sentence_len < 50") # Limit size to approx. 75th quantile
      .drop("sentence_len")
  )
  return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Extracting the Lemmas from our Data

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType 

@pandas_udf("string")
def lemmatize(txt: pd.Series) -> pd.Series:

  import nltk
  from nltk.corpus import stopwords
  from nltk.stem import WordNetLemmatizer
  from nltk.tokenize import RegexpTokenizer

  nltk.download('wordnet')
  nltk.download('omw-1.4')
  nltk.download('stopwords')
  nltk.download('punkt')

  lemmatizer = WordNetLemmatizer()
  lemmas_list = []

  for item in txt:

    #Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(item)

    #Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    #Lemmatize
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    lemmas_list.append(' '.join(lemmas))

  return pd.Series(lemmas_list)
  
df.repartition(20)
df.count()

df_lemmas = df.withColumn("lemmas", lemmatize("question_en"))
display(df_lemmas)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Visualizing most common lemmas in the dataset with a Wordcloud

# COMMAND ----------

!pip install -q wordcloud

# COMMAND ----------

from wordcloud import WordCloud
from matplotlib import pyplot as plt
import itertools

%config InlineBackend.figure_format='retina'

list_lemmas = list(itertools.chain.from_iterable(df_lemmas.select('lemmas').toLocalIterator()))
list_lemmas = sum([lemma.split(" ") for lemma in list_lemmas], [])
text = " ".join(list_lemmas)

# lower max_font_size
wordcloud = WordCloud(width = 1600, height = 800).generate(text)
plt.figure(figsize=(20,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# COMMAND ----------

# DBTITLE 1,Removing Outliers
df_lemmas = remove_outliers(df_lemmas)
df_lemmas.write.saveAsTable(name = "insuranceqa.train", mode = "overwrite", mergeSchema = True)

# COMMAND ----------

# DBTITLE 1,Extracting Lemmas from our Validation Set
df_valid = spark.sql("select * from insuranceqa.valid")
df_valid_lemmas = df_valid.withColumn("lemmas", lemmatize("question_en"))
df_valid_lemmas.write.saveAsTable(name = "insuranceqa.valid", mode = "overwrite", mergeSchema = True)
