from training.tasks.growing_2d import Growing2DTask
from training.tasks.growing_rf import GrowingRFTask
from training.tasks.growing_voxel import GrowingVoxelTask
from training.tasks.meshnca import MeshNCATask
from training.tasks.texture_2d import Texture2DTask
from training.tasks.texture_3d import Texture3DTask


def get_task(task_name: str, config: dict, logger):
    tasks = {
        "growing_2d": Growing2DTask,
        "texture_2d": Texture2DTask,
        "growing_rf": GrowingRFTask,
        "growing_voxel": GrowingVoxelTask,
        "texture_3d": Texture3DTask,
        "meshnca": MeshNCATask,
    }
    if task_name not in tasks:
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {sorted(tasks)}")
    return tasks[task_name](config, logger)

