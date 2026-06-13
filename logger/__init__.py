from .base import BaseLogger

def get_logger_class(backend: str) -> type[BaseLogger]:
    if backend == 'mlflow':
        from .mlflow import MLflowLogger
        return MLflowLogger
    if backend == 'wandb':
        from .wandb import WandbLogger
        return WandbLogger
    if backend == 'tensorboard':
        from .tensorboard import TensorboardLogger
        return TensorboardLogger
    if backend == 'disabled':
        from .disabled import DisabledLogger
        return DisabledLogger
    if backend == 'local':
        from .local_logger import LocalLogger
        return LocalLogger
    raise NotImplementedError()

def load_checkpoint(path: str, device: str = 'cpu'):
    import torch
    if path.startswith('s3://'):
        # Download via mlflow
        import mlflow
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=path
        )
        return torch.load(local_path, map_location=device)
    return torch.load(path, map_location=device)

def load_config(path: str) -> dict:
    import yaml
    if path.startswith('s3://'):
        # Download via mlflow
        import mlflow
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=path
        )
        with open(local_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg
