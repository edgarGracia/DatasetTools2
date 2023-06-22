from __future__ import annotations

from enum import Enum, unique
from typing import Union, Tuple, Type
from copy import deepcopy

import numpy as np

int_or_float = Union[int, float]


@unique
class CoordinatesType(Enum):
    RELATIVE = "RELATIVE"
    ABSOLUTE = "ABSOLUTE"


class BaseBoundingBox:
    """Base class for bounding boxes.
    """

    def __int__(
        self,
        x1: int_or_float,
        y1: int_or_float,
        x2: int_or_float,
        y2: int_or_float,
        coords_type: CoordinatesType = CoordinatesType.ABSOLUTE
    ):
        raise NotImplementedError

    @property
    def numpy(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def size(self) -> Tuple[int_or_float, int_or_float]:
        raise NotImplementedError

    @property
    def width(self) -> int_or_float:
        raise NotImplementedError

    @property
    def height(self) -> int_or_float:
        raise NotImplementedError

    @property
    def cx(self) -> int_or_float:
        raise NotADirectoryError

    @property
    def cy(self) -> int_or_float:
        raise NotADirectoryError

    @property
    def xmin(self) -> int_or_float:
        raise NotADirectoryError

    @property
    def ymin(self) -> int_or_float:
        raise NotADirectoryError

    @property
    def xmax(self) -> int_or_float:
        raise NotADirectoryError

    @property
    def ymax(self) -> int_or_float:
        raise NotADirectoryError

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_absolute(
        self,
        image_width: int,
        image_height: int
    ) -> Type[BaseBoundingBox]:
        raise NotImplementedError

    def to_relative(
        self,
        image_width: int,
        image_height: int
    ) -> Type[BaseBoundingBox]:
        raise NotImplementedError

    @property
    def is_valid(self) -> bool:
        if (self.xmin < 0 or self.ymin < 0 or self.xmin >= self.xmax or
            self.ymin >= self.ymax):
            return False
        if self.coords_type is CoordinatesType.RELATIVE:
            if self.xmin > 1 or self.xmax > 1 or self.ymin > 1 or self.ymax > 1:
                return False
        return True

    @classmethod
    def from_xyxy(
        cls,
        xmin: int_or_float,
        ymin: int_or_float,
        xmax: int_or_float,
        ymax: int_or_float,
        coords_type: CoordinatesType = CoordinatesType.ABSOLUTE
    ):
        raise NotImplementedError

    def to(self, cls: Type[BaseBoundingBox]) -> Type[BaseBoundingBox]:
        cls.from_xyxy(
            xmin=self.xmin,
            ymin=self.ymin,
            xmax=self.xmax,
            ymax=self.ymax,
            coords_type=self.coords_type
        )

    def scale(
        self,
        fx: float = 1,
        fy: float = 1,
        from_center: bool = False
    ) -> Type[BaseBoundingBox]:
        """Scale the bounding box by a factor.

        Args:
            fx (float, optional): Horizontal scale factor. Defaults to 1.
            fy (float, optional): Vertical scale factor. Defaults to 1.
            from_center (bool, optional): If the scale should be done from
                the center of the box. Defaults to False.

        Returns:
            Type[BaseBoundingBox]: A scaled copy of the bounding box.
        """
        if from_center:
            new_w = self.width * fx
            new_h = self.height * fy
            xmin = self.cx - (new_w/2)
            ymin = self.cy - (new_h/2)
            xmax = self.cx + (new_w/2)
            ymax = self.cy + (new_h/2)
        else:
            xmin = fx * self.xmin
            ymin = fy * self.ymin
            xmax = fx * self.xmax
            ymax = fy * self.ymax
        return type(self).from_xyxy(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            coords_type=self.coords_type
        )

    def resize_to_image(
        self,
        new_width: int,
        new_height: int,
        src_width: int,
        src_height: int
    ) -> Type[BaseBoundingBox]:
        """Resize a bounding box to a new image size. The box should be in
        absolute coordinates.

        Args:
            new_width (int): New image width.
            new_height (int): New image height.
            src_width (int): Source image width.
            src_height (int): Source image height.

        Raises:
            ValueError: If the coordinates are relative.

        Returns:
            Type[BaseBoundingBox]: A new resized bounding box object.
        """
        if self.coords_type is CoordinatesType.RELATIVE:
            raise ValueError("Can not resize to image a relative bounding box")
        fx = new_width / src_width
        fy = new_height / src_height
        return self.scale(fx=fx, fy=fy)

    def to(self, cls: Type[BaseBoundingBox]) -> Type[BaseBoundingBox]:
        raise NotImplementedError

    @classmethod
    def from_xyxy(self, xmin: ):
        raise NotImplementedError
    
    def copy(self) -> Type[BaseBoundingBox]:
        return deepcopy(self)

    def __repr__(self) -> str:
        raise NotImplementedError
    
    def __str__(self) -> str:
        return repr(self)

    @property
    def size(self) -> Tuple[int_or_float, int_or_float]:
        return (self.width, self.height)

    def _cast_value(self, value: int_or_float) -> int_or_float:
        if self.coords_type is CoordinatesType.ABSOLUTE:
            return int(value)
        return float(value)

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        box = self.to_absolute(w, h)
        return image[box.ymin:box.ymax, box.xmin:box.xmax]


class BoundingBoxXYXY(BaseBoundingBox):
    """Bounding box from the top-left point (xmin, ymin) and bottom-right point
    (xmax, ymax).
    """

    def __init__(
        self,
        xmin: int_or_float,
        ymin: int_or_float,
        xmax: int_or_float,
        ymax: int_or_float,
        coords_type: CoordinatesType = CoordinatesType.ABSOLUTE
    ):
        self.coords_type = coords_type
        self._xmin = self._cast_value(xmin)
        self._ymin = self._cast_value(ymin)
        self._xmax = self._cast_value(xmax)
        self._ymax = self._cast_value(ymax)

    @property
    def numpy(self) -> np.ndarray:
        return np.array(
            [self._xmin, self._ymin, self._xmax, self._ymax],
            dtype=("int" if self.coords_type is CoordinatesType.ABSOLUTE
                   else "float")
        )

    @property
    def width(self) -> int_or_float:
        return self._xmax - self._xmin

    @property
    def height(self) -> int_or_float:
        return self._ymax - self._ymin

    @property
    def cx(self) -> int_or_float:
        return self._cast_value(
            (self.xmin + self.xmax)/2
        )

    @property
    def cy(self) -> int_or_float:
        return self._cast_value(
            (self.ymin + self.ymax)/2
        )

    @property
    def xmin(self) -> int_or_float:
        return self._xmin

    @property
    def ymin(self) -> int_or_float:
        return self._ymin

    @property
    def xmax(self) -> int_or_float:
        return self._xmax

    @property
    def ymax(self) -> int_or_float:
        return self._ymax

    def to_absolute(
        self,
        image_width: int,
        image_height: int
    ) -> BoundingBoxXYXY:
        if self.coords_type is CoordinatesType.ABSOLUTE:
            return self.copy()
        return BoundingBoxXYXY(
            xmin=np.clip(round(self._xmin * (image_width-1)), 0, image_width),
            ymin=np.clip(round(self._ymin * (image_height-1)), 0, image_height),
            xmax=np.clip(round(self._xmax * (image_width-1)), 0, image_width),
            ymax=np.clip(round(self._ymax * (image_height-1)), 0, image_height),
            coords_type=CoordinatesType.ABSOLUTE
        )

    def to_relative(
        self,
        image_width: int,
        image_height: int
    ) -> BoundingBoxXYXY:
        if self.coords_type is CoordinatesType.RELATIVE:
            return self.copy()
        return BoundingBoxXYXY(
            xmin=np.clip(self._xmin / image_width, 0, 1),
            ymin=np.clip(self._ymin / image_height, 0, 1),
            xmax=np.clip(self._xmax / image_width, 0, 1),
            ymax=np.clip(self._ymax / image_height, 0, 1),
            coords_type=CoordinatesType.RELATIVE
        )

    @classmethod
    def from_xyxy(
        cls,
        xmin: int_or_float,
        ymin: int_or_float,
        xmax: int_or_float,
        ymax: int_or_float,
        coords_type: CoordinatesType = CoordinatesType.ABSOLUTE
    ):
        return BoundingBoxXYXY(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            coords_type=coords_type
        )
        
    def __repr__(self) -> str:
        return f"BoundingBoxXYXY(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax}, coords_type={self.coords_type})"


class BoundingBoxCXCYWH(BaseBoundingBox):
    """Bounding box from the center point (cx, cy) and width and height (w, h).
    """

    def __init__(
        self,
        cx: int_or_float,
        cy: int_or_float,
        w: int_or_float,
        h: int_or_float,
        coords_type: CoordinatesType = CoordinatesType.ABSOLUTE
    ):
        self.coords_type = coords_type
        self._cx = self._cast_value(cx)
        self._cy = self._cast_value(cy)
        self._w = self._cast_value(w)
        self._h = self._cast_value(h)

    @property
    def numpy(self) -> np.ndarray:
        return np.array(
            [self._cx, self._cy, self._w, self._h],
            dtype=("int" if self.coords_type is CoordinatesType.ABSOLUTE
                   else "float")
        )

    @property
    def width(self) -> int_or_float:
        return self._w

    @property
    def height(self) -> int_or_float:
        return self._h

    @property
    def cx(self) -> int_or_float:
        return self._cx

    @property
    def cy(self) -> int_or_float:
        return self.cy

    @property
    def xmin(self) -> int_or_float:
        return self._cast_value(
            self._cx - (self.w / 2)
        )

    @property
    def ymin(self) -> int_or_float:
        return self._cast_value(
            self._cy - (self.h / 2)
        )

    @property
    def xmax(self) -> int_or_float:
        return self._cast_value(
            self._cx + (self.w / 2)
        )

    @property
    def ymax(self) -> int_or_float:
        return self._cast_value(
            self._cy + (self.h / 2)
        )

    def to_absolute(
        self,
        image_width: int,
        image_height: int
    ) -> BoundingBoxCXCYWH:
        if self.coords_type is CoordinatesType.ABSOLUTE:
            return self.copy()
        return BoundingBoxCXCYWH(
            cx=np.clip(round(self._cx * (image_width-1)), 0, image_width),
            cy=np.clip(round(self._cy * (image_height-1)), 0, image_height),
            w=np.clip(round(self._w * (image_width-1)), 0, image_width),
            h=np.clip(round(self._h * (image_height-1)), 0, image_height),
            coords_type=CoordinatesType.ABSOLUTE
        )

    def to_relative(
        self,
        image_width: int,
        image_height: int
    ) -> BoundingBoxCXCYWH:
        if self.coords_type is CoordinatesType.RELATIVE:
            return self.copy()
        return BoundingBoxCXCYWH(
            cx=np.clip(self._cx / image_width, 0, 1),
            cy=np.clip(self._cy / image_height, 0, 1),
            w=np.clip(self._w / image_width, 0, 1),
            h=np.clip(self._h / image_height, 0, 1),
            coords_type=CoordinatesType.RELATIVE
        )

    @classmethod
    def from_xyxy(
        cls,
        xmin: int_or_float,
        ymin: int_or_float,
        xmax: int_or_float,
        ymax: int_or_float,
        coords_type: CoordinatesType = CoordinatesType.ABSOLUTE
    ):
        return BoundingBoxCXCYWH(
            cx=(xmin + xmax) / 2,
            cy=(ymin + ymax) / 2,
            w=(xmax - xmin),
            h=(ymax - ymin),
            coords_type=coords_type
        )
        
    def __repr__(self) -> str:
        return f"BoundingBoxCXCYWH(cx={self._cx}, cy={self._cy}, w={self._w}, h={self._h}, coords_type={self.coords_type})"


class BoundingBoxX1Y1WH(BaseBoundingBox):
    """Bounding box from the top-left point (xmin, ymin) and width and height
    (w, h).
    """

    def __init__(
        self,
        xmin: int_or_float,
        ymin: int_or_float,
        w: int_or_float,
        h: int_or_float,
        coords_type: CoordinatesType = CoordinatesType.ABSOLUTE
    ):
        self.coords_type = coords_type
        self._xmin = self._cast_value(xmin)
        self._ymin = self._cast_value(ymin)
        self._w = self._cast_value(w)
        self._h = self._cast_value(h)

    @property
    def numpy(self) -> np.ndarray:
        return np.array(
            [self._xmin, self._ymin, self._w, self._h],
            dtype=("int" if self.coords_type is CoordinatesType.ABSOLUTE
                   else "float")
        )

    @property
    def width(self) -> int_or_float:
        return self._w

    @property
    def height(self) -> int_or_float:
        return self._h

    @property
    def cx(self) -> int_or_float:
        return self._cast_value(
            self._xmin + (self._w/2)
        )

    @property
    def cy(self) -> int_or_float:
        return self._cast_value(
            self._ymin + (self._h/2)
        )

    @property
    def xmin(self) -> int_or_float:
        return self._xmin

    @property
    def ymin(self) -> int_or_float:
        return self._ymin

    @property
    def xmax(self) -> int_or_float:
        return self._xmin + self._w

    @property
    def ymax(self) -> int_or_float:
        return self._ymin + self._h

    def to_absolute(
        self,
        image_width: int,
        image_height: int
    ) -> BoundingBoxX1Y1WH:
        if self.coords_type is CoordinatesType.ABSOLUTE:
            return self.copy()
        return BoundingBoxX1Y1WH(
            xmin=np.clip(round(self._xmin * (image_width-1)), 0, image_width),
            ymin=np.clip(round(self._ymin * (image_height-1)), 0, image_height),
            w=np.clip(round(self._w * (image_width-1)), 0, image_width),
            h=np.clip(round(self._h * (image_height-1)), 0, image_height),
            coords_type=CoordinatesType.ABSOLUTE
        )

    def to_relative(
        self,
        image_width: int,
        image_height: int
    ) -> BoundingBoxX1Y1WH:
        if self.coords_type is CoordinatesType.RELATIVE:
            return self.copy()
        return BoundingBoxX1Y1WH(
            xmin=np.clip(self._xmin / image_width, 0, 1),
            ymin=np.clip(self._ymin / image_height, 0, 1),
            w=np.clip(self._w / image_width, 0, 1),
            h=np.clip(self._h / image_height, 0, 1),
            coords_type=CoordinatesType.RELATIVE
        )

    @classmethod
    def from_xyxy(
        cls,
        xmin: int_or_float,
        ymin: int_or_float,
        xmax: int_or_float,
        ymax: int_or_float,
        coords_type: CoordinatesType = CoordinatesType.ABSOLUTE
    ):
        return BoundingBoxX1Y1WH(
            xmin=xmin,
            ymin=ymin,
            w=(xmax - xmin),
            h=(ymax - ymin),
            coords_type=coords_type
        )
        
    def __repr__(self) -> str:
        return f"BoundingBoxX1Y1WH(xmin={self._xmin}, ymin={self._ymin}, w={self._w}, h={self._h}, coords_type={self.coords_type})"
