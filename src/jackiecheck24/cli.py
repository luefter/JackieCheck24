from lightning.pytorch.cli import LightningCLI

from jackiecheck24.callbacks.cli import MLFlowSaveModel


class ConvenientLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MLFlowSaveModel, "after_fit")
        parser.set_defaults({"after_fit.save_model": False})