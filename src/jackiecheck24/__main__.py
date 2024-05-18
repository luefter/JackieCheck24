import lightning

from jackiecheck24.callbacks.cli import MLFlowSaveConfigCallback, MLFlowSaveModel
from jackiecheck24.cli import ConvenientLightningCLI
from jackiecheck24.data.dataset import MovieLensDataModule
from jackiecheck24.model.modules import MovieRecommender

lightning.seed_everything(0)


def cli_main():
    cli = ConvenientLightningCLI(
        model_class=MovieRecommender,
        datamodule_class=MovieLensDataModule,
        save_config_callback=MLFlowSaveConfigCallback,
    )


if __name__ == "__main__":
    cli_main()
