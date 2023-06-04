from typing import List, Optional

from omegaconf import DictConfig

from DatasetTools.Config.config import get_cfg
from DatasetTools.Structures.image import Image


class BaseParser:

    def __init__(self, cfg: Optional[DictConfig] = None):
        self.cfg = get_cfg() if cfg is None else cfg
        self._images: List[Image] = []

    def load(self) -> None:
        """Parse a dataset.
        """
        raise NotImplementedError

    def images(self) -> List[Image]:
        """Get a list with the images of the dataset
        """
        return self._images
