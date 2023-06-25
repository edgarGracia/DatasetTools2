from enum import Enum, unique


@unique
class ColorSource(Enum):
    SOLID = "SOLID"
    LABEL = "LABEL"
    INSTANCE = "INSTANCE"
    KEYPOINTS = "KEYPOINTS"


@unique
class RelativePosition(Enum):
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


@unique
class CoordinatesType(Enum):
    RELATIVE = "RELATIVE"
    ABSOLUTE = "ABSOLUTE"
