from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path

from DatasetTools.Structures.instance import Instance


@dataclass
class Image:
    path: Path
    width: Optional[int] = None
    height: Optional[int] = None
    id: Optional[int] = None
    annotations: Optional[List[Instance]] = field(default_factory=list)
    meta: Optional[dict] = None
