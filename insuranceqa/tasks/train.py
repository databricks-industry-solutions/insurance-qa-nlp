from insuranceqa.datasets.insuranceqa import InsuranceDataset
from insuranceqa.models.distilbert_insuranceqa import LitModel
from insuranceqa.common import Task
from torch.utils.data import DataLoader
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging


class TrainTask(Task):
    def _get_data_loaders(self):
        suffix = self.conf["input"].get("suffix", "silver")
        train_split = f'{self.conf["input"].get("train")}_{suffix}'
        valid_split = f'{self.conf["input"].get("valid")}_{suffix}'
        test_split = f'{self.conf["input"].get("test")}_{suffix}'
        num_workers = self.conf["num_workers"]
        batch_size = self.conf["batch_size"]

        train_data = InsuranceDataset(spark=self.spark, split=train_split)
        valid_data = InsuranceDataset(spark=self.spark, split=valid_split)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_dataloader, valid_dataloader

    def _train(self):
        train_dataloader, valid_dataloader = self._get_data_loaders()
        lit_model = LitModel()

        max_epochs = self.conf.get("max_epochs", None)
        max_steps = self.conf.get("max_steps", None)
        if not max_steps:
            early_stop_callbacks = [
                EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, verbose=True, mode="min")
            ]
        else:
            early_stop_callbacks = None
        tracking_uri = self.conf.get("tracking_uri", "databricks")
        accelerator = self.conf.get("accelerator", "cpu")
        devices = self.conf.get("devices", None)
        experiment_name = self.conf["experiment_name"]
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="torch") as run:

            lit_model = LitModel()
            run_id = run.info.run_id
            mlf_logger = MLFlowLogger(
                experiment_name=experiment_name,
                run_id=run_id,
                tracking_uri=tracking_uri
            )
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                max_steps=max_steps,
                logger=mlf_logger,
                accelerator=accelerator,
                devices=devices,
                callbacks=early_stop_callbacks,
            )

            trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

            logging.info("Logging and registering model")
            model_info = mlflow.pytorch.log_model(lit_model, artifact_path="model")

            if tracking_uri == "databricks":
                model_version = mlflow.register_model(model_uri=model_info.model_uri, name="distilbert_insuranceqa")
                logging.info(f"Registered Model Version: {model_version}")

    def launch(self):
        self.logger.info("Launching ML training task")
        self._train()
        self.logger.info("ML training task finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = TrainTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
