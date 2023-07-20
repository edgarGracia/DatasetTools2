from typing import List

from DatasetTools.structures.sample import Sample


class BaseParser:

    def load(self) -> None:
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
