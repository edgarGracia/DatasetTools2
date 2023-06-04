from omegaconf import OmegaConf
from omegaconf import DictConfig
from copy import deepcopy

from DatasetTools.Config import defaults


def get_cfg() -> DictConfig:
    """Return the default configuration.
    """
    return defaults.cfg


def update_copy(cfg: DictConfig, values: dict) -> DictConfig:
    cfg = deepcopy(cfg)
    for k, v in values.items():
        OmegaConf.update(cfg, k, v)
    return cfg
