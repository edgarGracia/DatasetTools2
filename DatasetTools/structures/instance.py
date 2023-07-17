from __future__ import annotations
from typing import Optional, Dict, Type
from dataclasses import dataclass, field
from copy import deepcopy

from DatasetTools.structures.bounding_box import BaseBoundingBox
from DatasetTools.structures.mask import Mask


@dataclass
class Instance:
    bounding_box: Optional[Type[BaseBoundingBox]] = None
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

    @property
    def dict(self) -> dict:
        return {
            "bounding_box": self.bounding_box,
            "label": self.label,
            "label_id": self.label_id,
            "id": self.id,
            "mask": self.mask,
            "confidence": self.confidence,
            "text": self.text,
            "extras": self.extras,
        }
