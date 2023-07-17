from pathlib import Path
from typing import List, Optional

from DatasetTools.structures.instance import Instance
from DatasetTools.utils.utils import path_or_str


class Image:

    def __init__(
        self,
        path: path_or_str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        id: Optional[int] = None,
        annotations: Optional[List[Instance]] = None,
        meta: Optional[dict] = None
    ):
        self._path = path
        self.width = width
        self.height = height
        self.id = id
        self.annotations = [] if annotations is None else annotations
        self.meta = meta

    @property
    def path(self) -> Path:
        return Path(self._path)
