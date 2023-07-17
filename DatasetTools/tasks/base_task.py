from typing import Type

from omegaconf import DictConfig

from DatasetTools.data_parsers.base_parser import BaseParser


class BaseTask:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def run(self, parser: Type[BaseParser]):
        raise NotImplementedError
