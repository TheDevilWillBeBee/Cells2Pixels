import os
import datetime
from typing import Any
import numpy as np
from PIL.Image import Image as PILImage
from torch.utils.tensorboard import SummaryWriter

from .base import BaseLogger, ImageDataType

class TensorboardLogger(BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = kwargs.get('log_dir', 'runs')
        pass

    def start_run(self, run_name: str = None):
        """Initializes a new SummaryWriter for the run."""
        if self.writer:
            self.writer.close()
        
        self.writer: SummaryWriter = None
        self.run_name: str = None
        self.current_run_dir: str = None

        # TensorBoard automatically creates a timestamped subdirectory if the run_name isn't explicit
        # We handle this manually to ensure unique, predictable run directories
        if run_name:
            self.run_name = run_name
        else:
            self.run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.current_run_dir = os.path.join(self.log_dir, self.run_name)
        self.writer = SummaryWriter(log_dir=self.current_run_dir)
        print(f"Starting TensorBoard run: {self.run_name} at directory {self.current_run_dir}")

    def end_run(self, status: str = "FINISHED"):
        """Closes the SummaryWriter to flush any pending data."""
        if self.writer:
            self.writer.flush()
            self.writer.close()
            print(f"Ended TensorBoard run with status: {status}")
            self.writer = None
            self.run_name = None
            self.current_run_dir = None

    def log_param(self, key: str, value: Any):
        """Logs a hyperparameter as a text summary in TensorBoard."""
        if self.writer:
            # TensorBoard doesn't have a direct "log_param", so we use add_text
            # A common convention is to log parameters in a text dashboard
            self.writer.add_text("hyperparameters", f"**{key}**: {value}")

    def log_metric(self, key: str, value: float, step: int = None):
        """Logs a scalar metric."""
        if self.writer:
            self.writer.add_scalar(key, value, global_step=step)

    def log_metrics(self, kv: dict, step: int = None):
        self.writer.add_scalars('', kv, global_step=step)

    def log_image(self, image: ImageDataType, key: str, caption: str = None, step: int = None):
        """Logs an image, converting it to a NumPy array if necessary."""
        if self.writer:
            # Handle different image input types
            if isinstance(image, PILImage):
                image = np.array(image)
            elif isinstance(image, str):
                # If image is a file path, load it using Pillow
                try:
                    from PIL import Image
                    image = np.array(Image.open(image))
                except ImportError:
                    print("Pillow is not installed. Cannot log image from file path.")
                    return
            
            # TensorBoard's `add_image` method supports captions via the `text` parameter in a roundabout way
            # Here we just log the caption as text for simplicity
            if caption:
                self.writer.add_text(f"{key}_caption", caption, global_step=step)

            # Log the image
            # `add_image` expects the format [N, H, W, C] for images
            self.writer.add_image(key, image, global_step=step, dataformats="HWC")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Logs an artifact by copying it to the run directory."""
        if self.writer and self.current_run_dir:
            import shutil
            # Ensure the destination path exists
            if artifact_path:
                dest_dir = os.path.join(self.current_run_dir, artifact_path)
            else:
                dest_dir = self.current_run_dir
            
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(local_path, dest_dir)
            print(f"Logged artifact {local_path} to {dest_dir}")

    def log_model(self, model: Any, model_name: str):
        print("Unsupported method. Skipped")

