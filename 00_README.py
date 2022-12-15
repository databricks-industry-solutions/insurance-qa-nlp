# Databricks notebook source
# MAGIC %md This notebook can be found at https://github.com/databricks-industry-solutions/insurance-qa-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC # Insurance Q&A Intent Classification with Databricks & Hugging Face
# MAGIC 
# MAGIC <img src="https://github.com/rafaelvp-db/dbx-insurance-qa-hugging-face/blob/master/img/header.png?raw=true" />
# MAGIC 
# MAGIC <hr />
# MAGIC 
# MAGIC **TLDR;** this repo contains code that showcases the process of:
# MAGIC * Ingesting data related to Insurance questions and answers ([Insurance QA Dataset](https://github.com/shuzi/insuranceQA)) into Delta Lake
# MAGIC * Basic cleaning and preprocessing
# MAGIC * Creating custom [PyTorch Lightning](https://www.pytorchlightning.ai/) `DataModule` and `LightningModule` to wrap, respectively, our dataset and our backbone model (`distilbert_en_uncased`)
# MAGIC * Training with multiple GPUs while logging desired metrics into MLflow and registering model assets into Databricks Model Registry
# MAGIC * Running inference both with single and multiple nodes
# MAGIC 
# MAGIC 
# MAGIC ## Additional Reference
# MAGIC 
# MAGIC 1. Minwei Feng, Bing Xiang, Michael R. Glass, Lidan Wang, Bowen Zhou. [Applying Deep Learning to Answer Selection: A Study and An Open Task](https://arxiv.org/abs/1508.01585)
# MAGIC 2. [Fine-tune Transformers Models with PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html)
# MAGIC 3. [PyTorch Lightning MLflow Logger](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.mlflow.html)

# COMMAND ----------


