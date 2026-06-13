import os
import json
import shutil
from datetime import datetime
from typing import Union, List
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseLogger, ImageDataType


class LocalLogger(BaseLogger):
    def __init__(self, experiments_dir: str = "experiments", run_dir: str = None, **kwargs):
        super().__init__(**kwargs)
        self.experiments_dir = experiments_dir
        self.run_dir = run_dir
        self.current_run_dir = None
        self.run_name = None
        self.metrics_log = []
        
        if self.run_dir is None:
            os.makedirs(self.experiments_dir, exist_ok=True)
    
    def start_run(self, run_name: str = None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        self.current_run_dir = self.run_dir or os.path.join(self.experiments_dir, run_name)
        
        # Create run directory and subdirectories
        os.makedirs(self.current_run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.current_run_dir, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(self.current_run_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.current_run_dir, "models"), exist_ok=True)
        
        self.metrics_log = []
        
        # Log run start time
        with open(os.path.join(self.current_run_dir, "run_info.json"), "w") as f:
            json.dump({
                "run_name": run_name,
                "start_time": datetime.now().isoformat(),
                "status": "RUNNING"
            }, f, indent=2)

    def log_param(self, key: str, value: any):
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        params_file = os.path.join(self.current_run_dir, "params.json")
        
        # Load existing params or create new
        params = {}
        if os.path.exists(params_file):
            with open(params_file, "r") as f:
                params = json.load(f)
        
        params[key] = value
        
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)

    def log_metric(self, key: str, value: float, step: int = None):
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        metric_entry = {
            "key": key,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        
        self.metrics_log.append(metric_entry)
        
        # Save metrics log
        with open(os.path.join(self.current_run_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics_log, f, indent=2)

    def log_metrics(self, kv: dict, step: int = None):
        for key, value in kv.items():
            self.log_metric(key, value, step)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        if artifact_path is None:
            artifact_path = os.path.basename(local_path)
        
        dest_path = os.path.join(self.current_run_dir, "artifacts", artifact_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if os.path.isfile(local_path):
            shutil.copy2(local_path, dest_path)
        elif os.path.isdir(local_path):
            shutil.copytree(local_path, dest_path, dirs_exist_ok=True)

    def log_model(self, model, model_name: str):
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        model_dir = os.path.join(self.current_run_dir, "models", model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to save model using common frameworks
        try:
            # PyTorch
            import torch
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
                return
        except ImportError:
            pass
        
        try:
            # TensorFlow/Keras
            if hasattr(model, 'save'):
                model.save(os.path.join(model_dir, "model"))
                return
        except:
            pass
        
        # Fallback: save model info
        with open(os.path.join(model_dir, "model_info.txt"), "w") as f:
            f.write(f"Model type: {type(model)}\n")
            f.write(f"Model string representation: {str(model)}\n")

    def log_image(self, image: ImageDataType, key: str, caption: str = None, step: int = None):
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Create filename
        step_suffix = f"_step_{step}" if step is not None else ""
        filename = f"{key}{step_suffix}.png"
        image_path = os.path.join(self.current_run_dir, "images", filename)
        
        # Convert and save image
        if isinstance(image, str):
            # If string path, copy the image
            shutil.copy2(image, image_path)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image, 'RGB')
            elif len(image.shape) == 2:
                pil_image = Image.fromarray(image, 'L')
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
            pil_image.save(image_path)
        elif isinstance(image, PILImage):
            image.save(image_path)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Log image metadata
        image_metadata = {
            "key": key,
            "filename": filename,
            "caption": caption,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(self.current_run_dir, "images_metadata.json")
        metadata_list = []
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata_list = json.load(f)
        
        metadata_list.append(image_metadata)
        
        with open(metadata_file, "w") as f:
            json.dump(metadata_list, f, indent=2)

    def end_run(self, status: str = "FINISHED"):
        if self.current_run_dir is None:
            raise RuntimeError("No active run to end.")
        
        # Update run info with end time and status
        run_info_file = os.path.join(self.current_run_dir, "run_info.json")
        with open(run_info_file, "r") as f:
            run_info = json.load(f)
        
        run_info["end_time"] = datetime.now().isoformat()
        run_info["status"] = status
        
        with open(run_info_file, "w") as f:
            json.dump(run_info, f, indent=2)
        
        self.current_run_dir = None
        self.run_name = None
