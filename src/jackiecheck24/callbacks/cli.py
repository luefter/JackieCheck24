import mlflow
from mlflow.models import infer_signature

import lightning as pl
from lightning import Trainer, LightningModule, Callback
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger


class MLFlowSaveConfigCallback(SaveConfigCallback):
    def __init__(
        self,
        parser,
        config,
        config_filename="config.yaml",
        overwrite=False,
        multifile=False,
    ):
        super().__init__(
            parser, config, config_filename, overwrite, multifile, save_to_log_dir=False
        )

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, MLFlowLogger):
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams({"config": config})


class MLFlowSaveModel(Callback):
    def __init__(self, model_name: str = "your_model_name", save_model: bool = False):
        super().__init__()
        self.model_input = None
        self.model_signature = None
        self.model_name = model_name
        self.save_model = save_model

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if self.model_input is None:
            self.model_input = batch

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.save_model:
            self.model_signature = infer_signature(
                {key: value.numpy() for key, value in self.model_input.items()},
                pl_module(self.model_input).detach().numpy(),
            )

            with mlflow.start_run(run_id=trainer.logger._run_id) as _:
                mlflow.pytorch.log_model(
                    pytorch_model=pl_module,
                    artifact_path=self.model_name,
                    registered_model_name=self.model_name,
                    signature=self.model_signature,
                )
