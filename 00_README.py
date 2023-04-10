# Databricks notebook source
# MAGIC %md This notebook can be found at https://github.com/databricks-industry-solutions/insurance-qa-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC <div >
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/solution-accelerator-logo.png"; width="50%">
# MAGIC </div>
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC While companies across industries have accelerated digital adoption, are insurers meeting the ever-changing demands from customers? As an insurer, are you spending most of your time and resources into creating business value?
# MAGIC 
# MAGIC Customer service is a vital part of the insurance business, for multiple business cases, from marketing, to customer retention and claims. By digitalizing these processes, insurers can seamlessly scale their operations and shift money and resources from operational processes to actual product and value creation.
# MAGIC 
# MAGIC This solutions accelerator is a head start on developing and deploying a machine learning solution to detect customer intents based on pieces of text from an Interactive Voice Response (IVR) stream or from a virtual agent.
# MAGIC 
# MAGIC **Authors**
# MAGIC 
# MAGIC * Rafael Piere ([rafael.pierre@databricks.com](mailto:rafael.pierre@databricks.com))

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2023]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |PyTorch|BSD License| https://github.com/pytorch/pytorch/blob/master/LICENSE| https://github.com/pytorch/pytorch/|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache 2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
# MAGIC |Transformers|Apache 2.0|https://github.com/huggingface/transformers/blob/main/LICENSE|https://github.com/huggingface/transformers/|
# MAGIC |Datasets|Apache 2.0|https://github.com/huggingface/datasets/blob/main/LICENSE|https://github.com/huggingface/datasets/|

# COMMAND ----------


