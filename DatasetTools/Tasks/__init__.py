from typing import Type

from omegaconf import DictConfig

from .visualization import BaseTask, Visualization

tasks = {
    "visualization": Visualization
}

def create_task(cfg: DictConfig) -> Type[BaseTask]:
    task = list(cfg.task.keys())[0]
    if task not in tasks:
        raise KeyError(f"{task} is not a valid task "
                       f"({list(tasks.keys())})")
    return tasks[task](cfg)
