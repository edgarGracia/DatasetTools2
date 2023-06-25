from typing import List, Optional

from omegaconf import DictConfig

from DatasetTools.structures.image import Image


class BaseParser:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._images: List[Image] = []

    def load(self) -> None:
        """Parse a dataset.
        """
        raise NotImplementedError

    def images(self) -> List[Image]:
        """Get a list with the images of the dataset
        """
        return self._images
