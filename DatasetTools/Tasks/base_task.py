from __future__ import annotations

import argparse
from typing import Optional, Type

from omegaconf import DictConfig

from DatasetTools.datasets.base_parser import BaseParser


class BaseTask:

    def run(self, parser: Type[BaseParser]):
        raise NotImplementedError

    @classmethod
    def add_sub_parser(cls, parent_parser: argparse.ArgumentParser):
        raise NotImplementedError

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        cfg: Optional[DictConfig]
    ) -> BaseTask:
        raise NotImplementedError
