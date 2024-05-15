import torch
import lightning
from lightning.pytorch.cli import LightningCLI

from jackiecheck24.callbacks.cli import MLFlowSaveConfigCallback, MLFlowSaveModel
from jackiecheck24.data.dataset import MovieLensDataModule
from jackiecheck24.model.modules import MovieRecommender

lightning.seed_everything(0)



class ConvenientLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MLFlowSaveModel, "after_fit")
        parser.set_defaults({"after_fit.save_model": False})


def cli_main():
    cli = ConvenientLightningCLI(
        model_class=MovieRecommender,
        datamodule_class=MovieLensDataModule,
        save_config_callback=MLFlowSaveConfigCallback,
    )


if __name__ == "__main__":
    cli_main()
