from omegaconf import OmegaConf
from omegaconf import DictConfig
from copy import deepcopy
from typing import List
from omegaconf import DictConfig
from omegaconf import OmegaConf
import ast


from DatasetTools.config import defaults


def get_cfg() -> DictConfig:
    """Return the default configuration.
    """
    return defaults.cfg


def update_copy(cfg: DictConfig, values: dict) -> DictConfig:
    """Copy and update a cfg with the provided values.

    Args:
        cfg (DictConfig): The configuration to update.
        values (dict): A dictionary with the values to update. The keys are
            dot-separated cfg key strings. e.g. ``DATASET.PARSER``

    Returns:
        DictConfig: A copy of the configuration with the updated values.
    """
    cfg = deepcopy(cfg)
    for k, v in values.items():
        OmegaConf.update(cfg, k, v)
    return cfg


def update_copy_str(cfg: DictConfig, update_list: List[str]) -> DictConfig:
    """Copy and update a cfg from a list of string key-values assignations.

    Args:
        cfg (DictConfig): The cfg to update.
        update_list (List[str]): A list of strings representing configuration
            updates. e.g. ["DATASET.PARSER='COCODataset'"]

    Returns:
        DictConfig: A copy of the cfg with the updated values.
    """
    cfg = deepcopy(cfg)
    for update_str in update_list:
        key, value = update_str.split("=")
        value = ast.literal_eval(value.strip())
        OmegaConf.update(cfg, key.strip(), value)
    return cfg
