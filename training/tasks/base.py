from pathlib import Path

import torch

from training.common import TestOptions, count_parameters


class BaseTask:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config["device"])

    def train(self) -> None:
        raise NotImplementedError

    def test(self, options: TestOptions) -> None:
        raise NotImplementedError

    def _log_counts(self, model: torch.nn.Module, siren: torch.nn.Module, model_name: str) -> None:
        print(f"{model_name} parameters: {count_parameters(model)}")
        print(f"Siren parameters: {count_parameters(siren)}")

    def _optimizer(self, parameters):
        train_cfg = self.config["train"]
        optimizer = torch.optim.Adam(parameters, lr=train_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_cfg["lr_decay_steps"],
            gamma=train_cfg["lr_decay_gamma"],
        )
        return optimizer, scheduler

    def _save_logged_image(self, key: str, image, step: int) -> None:
        self.logger.log_image(image, key=key, caption=key, step=step)

    def _output_dir(self, options: TestOptions) -> Path:
        output_dir = Path(options.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

