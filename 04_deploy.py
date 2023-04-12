# Databricks notebook source
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
# MAGIC ### Deploying a Real Time Model using Databricks Model Serving
# MAGIC <br />
# MAGIC 
# MAGIC * In this step, we will deploy the model that we have trained and registered as a real time model serving endpoint
# MAGIC * In a real world scenario, this would allow us to expose our model as a REST API, allowing for real time use cases such as IVR routing and chatbots

# COMMAND ----------

# MAGIC %pip install -q datasets

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup config
# MAGIC %run ./config/notebook-config

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
import requests
import json

client = MlflowClient()
model_name = config["model_name"]
endpoint_name = f"{model_name}_v2"
model_info = client.get_latest_versions(name = model_name, stages = ["Production"])[0]

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
api_endpoint = "api/2.0/serving-endpoints"

headers = {
    'Authorization': 'Bearer {}'.format(databricks_token)
  }

def get_endpoint(endpoint_name: str) -> bool:

  payload = {"name": endpoint_name}

  response = requests.get(
    f"{databricks_url}/{api_endpoint}",
    data = json.dumps(payload),
    headers = headers
  ).json()

  endpoint_exists = endpoint_name in [endpoint["name"] for endpoint in response["endpoints"]]
  return endpoint_exists


def create_endpoint(endpoint_name: str, model_name: str, model_version: int) -> dict:

  payload = {
    "name": endpoint_name,
    "config": {
    "served_models": [{
      "model_name": model_name,
      "model_version": f"{model_info.version}",
      "workload_size": "Small",
      "scale_to_zero_enabled": "true"
      }]
    }
  }

  if not get_endpoint(endpoint_name):
    print("Endpoint doesn't exist, creating...")
    response = requests.post(
      f"{databricks_url}/{api_endpoint}",
      data = json.dumps(payload),
      headers = headers
    ).json()
  else:
    print("Endpoint exists, updating...")
    update_payload = {}
    update_payload["served_models"] = payload["config"]["served_models"]
    response = requests.put(
      f"{databricks_url}/{api_endpoint}/{endpoint_name}/config",
      data = json.dumps(update_payload),
      headers = headers
    ).json()

  return response

create_endpoint(
  endpoint_name = endpoint_name,
  model_name = model_info.name,
  model_version = model_info.version
)

# COMMAND ----------

# DBTITLE 1,Checking model endpoint deployment state
import time
from IPython.display import clear_output

def check_endpoint_status(endpoint_name: str, max_retries: int = 1000, interval: int = 5) -> str:
  """Check the Model Serving deployment status at every time step defined with the interval parameters"""

  current_tries = 0

  while current_tries < max_retries:
    clear_output(wait = True)
    response_json = requests.get(
      f"{databricks_url}/{api_endpoint}",
      headers = headers
    ).json()
    endpoint_status = [
      endpoint for endpoint
      in response_json["endpoints"]
      if endpoint["name"] == endpoint_name
    ][0]
    current_state = endpoint_status["state"]["config_update"]

    if (current_state == "IN_PROGRESS"):
      print(f"Checking model deployment status, attempt {current_tries} of {max_retries} - current state: {current_state}")
    else:
      message = f"Model endpoint deployment result: {endpoint_status}"
      return message

    current_tries += 1
    time.sleep(interval)

check_endpoint_status(endpoint_name = endpoint_name)

# COMMAND ----------

# DBTITLE 1,Querying the endpoint through REST API
def test_prediction_endpoint(questions):
  endpoint_url = f"serving-endpoints/{endpoint_name}/invocations"
  payload = {"instances": questions}
  
  data_json = json.dumps(payload)
  print(data_json)
  headers["Content-Type"] = 'application/json'
  response = requests.request(
    method='POST',
    headers = headers,
    url = f"{databricks_url}/{endpoint_url}",
    data = data_json
  )
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

test_questions = [
  "my car broke, what should I do?",
  "what is my life insurance coverage?",
  "can you send me my health insurance cover?"
]
test_prediction_endpoint(test_questions)
