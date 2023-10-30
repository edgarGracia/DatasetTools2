from typing import List

from DatasetTools.structures.sample import Sample
from DatasetTools.utils.utils import path_or_str


class BaseParser:

    def load(self) -> None:
        raise NotImplementedError

    def save(self, samples: List[Sample], output_path: path_or_str):
        raise NotImplementedError

    @property
    def meta(self) -> dict:
        return {}

    @property
    def samples(self) -> List[Sample]:
        return []

    @property
    def labels(self) -> dict:
        return {}
