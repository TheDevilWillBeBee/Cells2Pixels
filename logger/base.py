from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np
from PIL.Image import Image as PILImage

ImageDataType = Union[np.ndarray, PILImage, str]


class BaseLogger(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self.log_loss_every_step = kwargs.get("log_loss_every_step", False)
        pass

    @abstractmethod
    def start_run(self, run_name: str = None):
        pass

    @abstractmethod
    def log_param(self, key: str, value: any):
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int = None):
        pass

    @abstractmethod
    def log_metrics(self, kv: dict, step: int = None):
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str = None):
        pass

    @abstractmethod
    def log_model(self, model, model_name: str):
        pass

    @abstractmethod
    def log_image(
        self, image: ImageDataType, key: str, caption: str = None, step: int = None
    ):
        pass

    @abstractmethod
    def end_run(self, status: str = "FINISHED"):
        pass
