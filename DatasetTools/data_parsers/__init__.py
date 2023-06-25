from typing import Type

from omegaconf import DictConfig

from .coco_parser import BaseParser, COCODataset

data_parsers = {
    "COCODataset": COCODataset
}


def create_parser(cfg: DictConfig) -> Type[BaseParser]:
    parser = cfg.dataset.parser
    if parser not in data_parsers:
        raise KeyError(f"{parser} is not a valid parser "
                       f"({list(data_parsers.keys())})")
    return data_parsers[parser](cfg)
