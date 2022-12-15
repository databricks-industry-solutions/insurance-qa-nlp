"""Helper class to download the Insurance QA dataset."""

import shutil
import requests
import logging
from pathlib import Path

DATASET_URL = "https://github.com/hailiang-wang/word2vec-get-started/archive/refs/heads/master.zip"
SUBFOLDER_PATH = "word2vec-get-started-master/corpus/insuranceqa/questions"


class InsuranceQA:
    def __init__(
        self,
        spark,
        dataset_url: str = DATASET_URL,
        filename: str = "master.zip",
        cache_dir: str = "/tmp",
        output_dir: str = "/tmp/insuranceqa",
        subfolder_path: str = SUBFOLDER_PATH,
        root_path: str = "file://"
    ):
        """Instantiates an InsuranceQA object.
        dataset_url: URL to the dataset.
        """

        self.spark = spark
        self.dataset_url = dataset_url
        self.filename = filename
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.subfolder_path = subfolder_path
        self.root_path = root_path

        self._download_and_extract()

    def _download_and_extract(self):

        filename = f"{self.cache_dir}/{self.filename}"
        response = requests.get(self.dataset_url)
        with open(filename, "wb") as file:
            file.write(response.content)
        logging.info(f"Downloaded {self.filename} to {filename}")
        format = filename.split(".")[-1]

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Unpacking {filename} into {self.output_dir}...")
        shutil.unpack_archive(filename, self.output_dir, format)
        logging.info("Success.")

    def ingest(self, database_name: str = "insuranceqa", split: str = "train", mode: str = "overwrite"):

        path = f"{self.root_path}/{self.output_dir}/{self.subfolder_path}/{split}.questions.txt"
        df = self.spark.read.csv(path, sep="\t", header=True)

        target_table = f"{database_name}.{split}"
        df = df.toDF("id", "topic_en", "topic_jp", "question_en", "question_jp").select("id", "topic_en", "question_en")

        self.spark.sql(f"create database if not exists {database_name}")
        df.write.saveAsTable(target_table, mode=mode)
