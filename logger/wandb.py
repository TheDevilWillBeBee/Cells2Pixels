import wandb
from .base import BaseLogger, ImageDataType

import os

class WandbLogger(BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.project = kwargs.get('project_name')
        self.api_key = kwargs.get("key")

    def start_run(self, run_name: str = None):
        wandb.login(key=self.api_key, relogin=True)
        wandb.init(project=self.project, name=run_name)

    def end_run(self, status: str = "FINISHED"):
        wandb.finish()

    def log_param(self, key: str, value: any):
        wandb.config.update({key: value})

    def log_metric(self, key: str, value: float, step: int = None):
        wandb.log({key: value}, step=step)

    def log_metrics(self, kv: dict, step: int = None):
        wandb.log(kv, step=step)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        artifact_name = artifact_path or wandb.util.to_snake_case(local_path)
        artifact = wandb.Artifact(name=artifact_name, type="file")
        if os.path.isdir(local_path):
            artifact.add_dir(local_path, name=artifact_name)
        elif os.path.isfile(local_path):
            artifact.add_file(local_path, name=artifact_name)
        else:
            print(f"Warning: {local_path} is not a file or directory. Skipping artifact logging.")
            return

        wandb.log_artifact(artifact)

    def log_model(self, model, model_name: str):
        # This is a simplified example; W&B has more robust model logging.
        wandb.sklearn.log_model(model, model_name)

    def log_image(self, image: ImageDataType, key: str, caption: str = None, step: int = None):
        # Create a wandb.Image object, which can handle various input types.
        wandb_image = wandb.Image(image, caption=caption)
        
        # Log the image using wandb.log, with the step parameter.
        if step is not None:
            wandb.log({key: wandb_image}, step=step)
        else:
            wandb.log({key: wandb_image})
