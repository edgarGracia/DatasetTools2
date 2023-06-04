from __future__ import annotations
from typing import Optional, Dict
from dataclasses import dataclass, field
from copy import deepcopy

from DatasetTools.Structures.bounding_box import BoundingBox
from DatasetTools.Structures.mask import Mask


@dataclass
class Instance:
    bounding_box: Optional[BoundingBox] = None
    label: Optional[str] = None
    label_id: Optional[int] = None
    id: Optional[int] = None
    mask: Optional[Mask] = None
    # keypoints: Optional[Keypoints] = None
    confidence: Optional[float] = None
    text: Optional[str] = None
    extras: Dict[str, any] = field(default_factory=dict)

    def copy(self) -> Instance:
        """Return a deep copy of the instance.
        """
        return deepcopy(self)
