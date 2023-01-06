![image](https://user-images.githubusercontent.com/86326159/206014015-a70e3581-e15c-4a10-95ef-36fd5a560717.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

## Insurance NLP Solution Accelerator
### Digitalization of the Claims Process using NLP, Delta and Hugging Face

<img src="https://raw.githubusercontent.com/rafaelvp-db/dbx-insurance-qa-hugging-face/master/img/header.png" width="100%"/>

While companies across industries have accelerated digital adoption, are insurers meeting the ever-changing demands from customers? As an insurer, are you spending most of your time and resources into creating business value?

**Customer service** is a vital part of the insurance business, for multiple business cases, from marketing, to customer retention and claims. By digitalizing these processes, insurers can seamlessly scale their operations and shift money and resources from operational processes to actual product and value creation.

This solutions accelerator is a head start on developing and deploying a machine learning solution to detect customer intents based on pieces of text from an Interactive Voice Response (IVR) stream or from a virtual agent.

### Target Solution

<img src="https://github.com/rafaelvp-db/dbx-insurance-qa-hugging-face/blob/master/img/Insurance(1).png?raw=true" />

### Authors

* Rafael Piere ([rafael.pierre@databricks.com](mailto:rafael.pierre@databricks.com))

___

Copyright Databricks, Inc. [2022]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

|Library Name|Library license | Library License URL | Library Source URL |
|---|---|---|---|
|PyTorch|BSD License| https://github.com/pytorch/pytorch/blob/master/LICENSE| https://github.com/pytorch/pytorch/|
|PyTorch Lightning|Apache-2.0 License |https://github.com/Lightning-AI/lightning/blob/master/LICENSE|https://github.com/Lightning-AI/lightning/|
|Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
|Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
|Transformers|Apache 2.0|https://github.com/huggingface/transformers/blob/main/LICENSE|https://github.com/huggingface/transformers/|

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
