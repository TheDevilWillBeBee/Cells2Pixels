from pathlib import PurePosixPath
from typing import Any


PBR_MAP_NAMES = {
    "albedo",
    "ao",
    "arm",
    "basecolor",
    "base_color",
    "color",
    "diffuse",
    "displacement",
    "height",
    "hra",
    "metallic",
    "metalness",
    "normal",
    "opacity",
    "roughness",
}


def append_target_name_to_experiment(config: dict[str, Any]) -> str | None:
    target_name = infer_target_name(config)
    if not target_name:
        return None

    suffix = f"-{target_name}"
    experiment_name = config["experiment_name"]
    if not experiment_name.endswith(suffix):
        config["experiment_name"] = f"{experiment_name}{suffix}"
    return target_name


def infer_target_name(config: dict[str, Any]) -> str | None:
    task = config["task"]
    loss_config = config.get("loss", {})

    if task == "growing_2d":
        return _stem(loss_config.get("image_loss_kwargs", {}).get("target_path"))

    if task in {"texture_2d", "texture_3d", "meshnca"}:
        target_paths = loss_config.get("appearance_loss_kwargs", {}).get("target_images_path")
        return _target_images_name(target_paths)

    if task == "growing_rf":
        return _name(loss_config.get("rf_loss_kwargs", {}).get("scene_path"))

    if task == "growing_voxel":
        voxel_kwargs = loss_config.get("voxel_loss_kwargs", {})
        mesh_name = _mesh_name(voxel_kwargs.get("mesh_path"))
        texture_name = _stem(voxel_kwargs.get("solid_texture_vol_path"))
        return "_".join(name for name in [mesh_name, texture_name] if name)

    return None


def _target_images_name(target_images_path) -> str | None:
    paths = _collect_paths(target_images_path)
    if not paths:
        return None

    parents = {path.parent.name for path in paths if path.parent.name}
    if len(paths) > 1 and len(parents) == 1:
        return next(iter(parents))

    if len(paths) == 1:
        path = paths[0]
        if path.stem.lower() in PBR_MAP_NAMES and path.parent.name:
            return path.parent.name
        return path.stem

    names = []
    for path in paths:
        if path.stem.lower() in PBR_MAP_NAMES and path.parent.name:
            names.append(path.parent.name)
        else:
            names.append(path.stem)
    return "_".join(dict.fromkeys(names))


def _collect_paths(value) -> list[PurePosixPath]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_as_path(value)]
    if isinstance(value, dict):
        return [_as_path(path) for path in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_path(path) for path in value]
    raise TypeError(f"Unsupported target_images_path type: {type(value)}")


def _mesh_name(path: str | None) -> str | None:
    if path is None:
        return None
    path = _as_path(path)
    if path.parent.name and path.parent.name == path.stem:
        return path.parent.name
    return path.stem


def _name(path: str | None) -> str | None:
    if path is None:
        return None
    path = _as_path(path)
    return path.name or path.stem


def _stem(path: str | None) -> str | None:
    if path is None:
        return None
    return _as_path(path).stem


def _as_path(path: str) -> PurePosixPath:
    return PurePosixPath(str(path).replace("\\", "/"))

