from insuranceqa.common import Task
from insuranceqa.data.ingest import InsuranceQA


class IngestionTask(Task):
    def _write_data(self):
        root_path = self.conf["input"].get("root_path", "file://")
        ds = InsuranceQA(
            self.spark,
            root_path = root_path
        )
        db = self.conf["output"].get("database", "insuranceqa")
        self.logger.info(f"Writing insuranceqa dataset to {db}")
        for split in ["train", "valid", "test"]:
            ds.ingest(database_name=db, split=split)
        self.logger.info("Dataset successfully ingested and written")

    def launch(self):
        self.logger.info("Launching ingestion task")
        self._write_data()
        self.logger.info("Ingestion task finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = IngestionTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
