from omegaconf import DictConfig


class BaseParser:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def load(self) -> None:
        """Parse a dataset.
        """
        raise NotImplementedError
