from .base import BaseLogger, ImageDataType


class DisabledLogger(BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start_run(self, run_name=None):
        pass

    def end_run(self, status: str = "FINISHED"):
        pass

    def log_param(self, key: str, value: any):
        pass

    def log_metric(self, key: str, value: float, step: int = None):
        pass

    def log_metrics(self, kv: dict, step: int = None):
        pass

    def log_artifact(self, local_path: str, artifact_path: str = None):
        pass

    def log_model(self, model, model_name: str):
        pass

    def log_image(
        self, image: ImageDataType, key: str, caption: str = None, step: int = None
    ):
        pass
