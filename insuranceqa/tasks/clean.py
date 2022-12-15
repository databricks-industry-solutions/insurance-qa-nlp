from insuranceqa.common import Task
from insuranceqa.data.clean import remove_outliers


class CleaningTask(Task):
    def _clean_data(self):
        db = self.conf["input"].get("database", "insuranceqa")
        suffix = self.conf["output"].get("suffix", "silver")
        filter = self.conf["filter"]
        for split in ["train", "valid", "test"]:
            remove_outliers(self.spark, database_name=db, split=split, suffix=suffix, filter=filter)
        self.logger.info("Successfully cleaned data")

    def launch(self):
        self.logger.info("Launching cleaning task")
        self._clean_data()
        self.logger.info("Cleaning task finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = CleaningTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
