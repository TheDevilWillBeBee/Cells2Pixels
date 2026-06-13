import mlflow
from .base import BaseLogger, ImageDataType

class MLflowLogger(BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        uri = None
        if (env_file := kwargs.get('env_file')):
            import os
            from dotenv import load_dotenv
            load_dotenv(env_file)
            uri = os.environ.get('MLFLOW_TRACKING_URI')
        if uri is None:
            uri = kwargs['uri']
        
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(kwargs['experiment'])

        self.log_system_metric = kwargs.get('log_system_metrics', False)
    
    def start_run(self, run_name = None):
        mlflow.start_run(
            run_name=run_name,
            log_system_metrics=self.log_system_metric
        )

    def end_run(self, status: str = "FINISHED"):
        mlflow.end_run(status=status)

    def log_param(self, key: str, value: any):
        mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int = None):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, kv: dict, step: int = None):
        mlflow.log_metrics(kv, step)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, model_name: str):
        mlflow.pytorch.log_model(model_name, python_model=model)

    def log_image(self, image: ImageDataType, key: str, caption: str = None, step: int = None):
        # MLflow doesn't support captions within log_image(),
        # so caption will be ignored.
        mlflow.log_image(image, artifact_file=f"{key}/{step:05d}.png")
