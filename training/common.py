import os
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(config: dict[str, Any], path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)


def ensure_experiment(config: dict[str, Any], config_path: str, overwrite: bool = False) -> Path:
    if "experiment_path" not in config:
        exp_name = sanitize_path_component(config["experiment_name"])
        config["experiment_path"] = str(Path("experiments") / exp_name)

    exp_path = Path(config["experiment_path"])
    if exp_path.exists() and (exp_path / "model.pth").exists() and not overwrite:
        raise FileExistsError(
            f"{exp_path} already contains model.pth. Pass --overwrite to train there anyway."
        )

    exp_path.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, exp_path / "config.yaml")
    if Path(config_path).resolve() != (exp_path / "config.yaml").resolve():
        try:
            shutil.copy2(config_path, exp_path / "source_config.yaml")
        except OSError:
            pass
    return exp_path


def sanitize_path_component(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def device_config(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg = dict(config)
    cfg["device"] = device
    return cfg


def precision_from_config(config: dict[str, Any]) -> torch.dtype:
    return torch.float32 if config.get("precision", "float32") == "float32" else torch.float16


def autocast_context(device: torch.device, precision: torch.dtype):
    enabled = device.type == "cuda" and precision == torch.float16
    if enabled:
        return torch.autocast(device_type=device.type, dtype=precision)
    return nullcontext()


def make_grad_scaler(device: torch.device, precision: torch.dtype):
    enabled = device.type == "cuda" and precision == torch.float16
    return torch.cuda.amp.GradScaler(enabled=enabled)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def normalize_model_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= p.grad.norm() + 1e-8


def save_checkpoint(config: dict[str, Any], model: torch.nn.Module, siren: torch.nn.Module, suffix: str = "") -> None:
    exp_path = Path(config["experiment_path"])
    model_name = f"model{suffix}.pth"
    siren_name = f"siren{suffix}.pth"
    torch.save(model.state_dict(), exp_path / model_name)
    torch.save(siren.state_dict(), exp_path / siren_name)


def load_checkpoint_pair(
    config: dict[str, Any],
    model: torch.nn.Module,
    siren: torch.nn.Module,
    directory: str | os.PathLike[str] | None = None,
    device: torch.device | str = "cpu",
    suffix: str = "",
) -> None:
    exp_path = Path(directory or config["experiment_path"])
    model_path = exp_path / f"model{suffix}.pth"
    siren_path = exp_path / f"siren{suffix}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    siren.load_state_dict(torch.load(siren_path, map_location=device, weights_only=True))


def load_graft_if_configured(
    config: dict[str, Any],
    section: str,
    model: torch.nn.Module,
    siren: torch.nn.Module,
    device: torch.device,
) -> int:
    graft_path = config.get(section, {}).get("graft_initialization")
    if graft_path is None:
        return 0

    graft_path = Path(graft_path)
    if (graft_path / "model.pth").exists():
        load_checkpoint_pair(config, model, siren, directory=graft_path, device=device)
        print(f"Loaded graft initialization from {graft_path}")
        return 0

    rep_start = 0
    for i in range(config["train"].get("num_repetitions", 1), 0, -1):
        if (graft_path / f"model_{i}.pth").exists():
            rep_start = i
            break

    if rep_start == 0:
        raise FileNotFoundError(f"No model.pth or model_N.pth found in graft path {graft_path}")

    load_checkpoint_pair(config, model, siren, directory=graft_path, device=device, suffix=f"_{rep_start}")
    print(f"Resuming from graft repetition {rep_start} in {graft_path}")
    return rep_start


@dataclass
class TestOptions:
    enabled: bool = False
    output_dir: str = "outputs"
    steps: int = 512
    video_frames: int = 240
    fps: float = 30.0
    save_image: bool = True
    save_video: bool = True
