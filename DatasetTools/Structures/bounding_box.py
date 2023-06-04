from __future__ import annotations

from enum import Enum, unique
from typing import Optional, Union, Tuple

import numpy as np


@unique
class BoundingBoxFormat(Enum):
    """Format of the bounding box coordinates.
    """
    XYXY = "XYXY"
    """xmin, ymin, xmax, ymax"""

    X1Y1WH = "X1Y1WH"
    """xmin, ymin, width, height"""

    CXCYWH = "CXCYWH"
    """center-x, center-y, width, height"""

    OTHER = "OTHER"


@unique
class CoordinatesType(Enum):
    RELATIVE = "RELATIVE"
    ABSOLUTE = "ABSOLUTE"


class BoundingBox:

    def __init__(
        self,
        x1: Union[float, int],
        y1: Union[float, int],
        x2: Union[float, int],
        y2: Union[float, int],
        format: BoundingBoxFormat = BoundingBoxFormat.XYXY,
        coords_type: CoordinatesType = CoordinatesType.ABSOLUTE
    ):
        """Create a new BoundingBox object from two set of points.

        Args:
            x1 (Union[float, int]): First x coordinate.
            y1 (Union[float, int]): First y coordinate.
            x2 (Union[float, int]): Second x coordinate.
            y2 (Union[float, int]): Second y coordinate.
            format (BoundingBoxFormat, optional): Bounding box points format.
                Defaults to BoundingBoxFormat.XYXY.
            coords_type (CoordinatesType, optional): Coordinates type.
                Defaults to CoordinatesType.ABSOLUTE.
        """
        BoundingBoxFormat.XYXY
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.format = format
        self.coords_type = coords_type

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Create a crop of an image using the bounding box.

        Args:
            image (np.ndarray): numpy image.

        Returns:
            np.ndarray: The cropped area of the image.
        """
        xmin, ymin, xmax, ymax = self.to(
            format=BoundingBoxFormat.XYXY,
            coords_type=CoordinatesType.ABSOLUTE,
            image_width=image.shape[1],
            image_height=image.shape[0]
        ).numpy().astype(int)
        return image[xmin:xmax, ymin:ymax]

    def numpy(self) -> np.ndarray:
        """Return the bounding box coordinates as a numpy array.

        Returns:
            np.ndarray: [x1, y1, x2, y2]
        """
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def to(
        self,
        format: Optional[BoundingBoxFormat] = None,
        coords_type: Optional[CoordinatesType] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ) -> BoundingBox:
        """Convert the bounding box to a different format.

        Args:
            format (Optional[BoundingBoxFormat], optional): A new bounding box
                format. If None, it will remain unchanged. Default to None.
            coords_type (Optional[CoordinatesType], optional): A new coordinates
                type. If None, it will remain unchanged. Default to None.
            image_width (Optional[int], optional): Image width. Required if the
                ``coords_type`` is changed. Defaults to None.
            image_height (Optional[int], optional): Image height. Required if
                the ``coords_type`` is changed. Defaults to None.

        Raises:
            NotImplementedError

        Returns:
            BoundingBox: A copy of the bounding box with the new format.
        """
        format = self.format if format is None else format
        coords_type = self.coords_type if coords_type is None else coords_type
        # Get the xmin, ymin, xmax, ymax
        if self.format is BoundingBoxFormat.XYXY:
            xmin = self.x1
            ymin = self.y1
            xmax = self.x2
            ymax = self.y2
        elif self.format is BoundingBoxFormat.X1Y1WH:
            xmin = self.x1
            ymin = self.y1
            xmax = xmin + self.x2
            ymax = ymin + self.y2
        elif self.format is BoundingBoxFormat.CXCYWH:
            xmin = self.x1 - (self.x2/2)
            ymin = self.y1 - (self.y2/2)
            xmax = self.x1 + (self.x2/2)
            ymax = self.y1 + (self.y2/2)
        else:
            raise NotImplementedError
        # Convert format
        if format is BoundingBoxFormat.XYXY:
            x1 = xmin
            y1 = ymin
            x2 = xmax
            y2 = ymax
        elif format is BoundingBoxFormat.X1Y1WH:
            x1 = xmin
            y1 = ymin
            x2 = xmax - xmin
            y2 = ymax - ymin
        elif format is BoundingBoxFormat.CXCYWH:
            x1 = (xmin + xmax)/2
            y1 = (ymin + ymax)/2
            x2 = xmax - xmin
            y2 = ymax - ymin
        else:
            raise NotImplementedError
        # Convert coords type
        if coords_type != self.coords_type:
            assert image_width is not None and image_height is not None
            if coords_type is CoordinatesType.ABSOLUTE:
                x1 = np.clip(round(x1 * (image_width-1)), 0, image_width)
                x2 = np.clip(round(x2 * (image_width-1)), 0, image_width)
                y1 = np.clip(round(y1 * (image_height-1)), 0, image_height)
                y2 = np.clip(round(y2 * (image_height-1)), 0, image_height)
            elif coords_type is CoordinatesType.RELATIVE:
                x1 = np.clip(x1 / image_width, 0, 1)
                x2 = np.clip(x2 / image_width, 0, 1)
                y1 = np.clip(y1 / image_height, 0, 1)
                y2 = np.clip(y2 / image_height, 0, 1)
        return BoundingBox(x1, y1, x2, y2,
                           format=format, coords_type=coords_type)

    def size(self) -> Tuple[Union[int, float], Union[int, float]]:
        """Return the bounding box size.

        Returns:
            Tuple[Union[int, float], Union[int, float]]: (width, height).
                Absolute or relative according to the box coordinates type.
        """
        box = self.to(BoundingBoxFormat.X1Y1WH)
        return (box.x2, box.y2)
    
    def width(self) -> Union[int, float]:
        """Returns:
            Union[int, float]: The bounding box width. Absolute or relative
                according to the box coordinates type.
        """
        return self.size()[0]

    def height(self) -> Union[int, float]:
        """Returns:
            Union[int, float]: The bounding box height. Absolute or relative
                according to the box coordinates type.
        """
        return self.size()[1]

    def is_valid(
        self,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ) -> bool:
        """Check if the coordinates of the bounding box are valid.

        Args:
            image_width (Optional[int], optional): Optional image width to
                check the coordinates. Defaults to None.
            image_height (Optional[int], optional): Optional image height to
                check the coordinates. Defaults to None.

        Returns:
            bool: True if it is valid.
        """
        size = self.size()
        if size[0] <= 0 or size[1] <= 0:
            return False
        if self.x1 < 0 or self.y1 < 0 or self.x2 < 0 or self.y2 < 0:
            return False
        if (self.coords_type is CoordinatesType.RELATIVE and 
            (self.x1 > 1 or self.y1 > 1 or self.x2 > 1 or self.y2 > 1)):
            return False
        if (self.coords_type is CoordinatesType.ABSOLUTE):
            box = self.to(BoundingBoxFormat.XYXY)
            if image_width is not None and box.x2 >= image_width:
                return False
            if image_height is not None and box.y2 >= image_height:
                return False
        return True

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"