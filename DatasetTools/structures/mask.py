from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
import pycocotools.mask as maskUtils


class Mask:

    def __init__(
        self,
        mask: Optional[np.ndarray] = None,
        rle: Optional[dict] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        area: Optional[int] = None
    ):
        """Create a Mask from a binary mask or an encoded rle.

        Args:
            mask (Optional[np.ndarray], optional): Binary mask of shape (H, W).
                Defaults to None.
            rle (Optional[dict], optional): Encoded rle mask. Defaults to None.
        """
        assert mask is not None or rle is not None
        self._rle = rle
        self._mask = mask
        self._width = width
        self._height = height
        self._area = area

    def _decode_rle(self, rle: Union[list, dict]) -> np.ndarray:
        # Parse rle/points
        if isinstance(rle, list):
            # List of points
            rle = maskUtils.frPyObjects(rle, self._height, self._width)
            rle = maskUtils.merge(rle)
        elif isinstance(rle["counts"], list):
            # uncompressed elw
            rle = maskUtils.frPyObjects(rle, self._height, self._width)
        return maskUtils.decode(rle)

    def numpy_mask(self) -> np.ndarray:
        """Get the mask as a numpy array.

        Returns:
            np.ndarray: A numpy mask of shape (H, W) of type ``bool``.
        """
        # TODO: check shape and type!
        if self._mask is None:
            mask = np.asfortranarray(self._decode_rle(self._rle).astype(bool))
            self._mask = mask
        return self._mask

    def rle(self) -> dict:
        """Get the mask an RLE.

        Returns:
            dict: The RLE
        """
        # TODO: Check return
        if self._rle is None:
            rle = maskUtils.encode(self._mask)
            self._rle = rle
        return self._rle

    def height(self) -> int:
        """Get the height of the mask.
        """
        # TODO: Test
        if self._height is None:
            if self._mask is None:
                self.numpy_mask()
            self.height = self._mask.shape[0]
        return self._height

    def width(self) -> int:
        """Get the width of the mask.
        """
        # TODO: Test
        if self._width is None:
            if self._mask is None:
                self.numpy_mask()
            self.width = self._mask.shape[1]
        return self._width

    def area(self) -> int:
        """Get the area of the mask.
        """
        # TODO: Check
        if self._area is None:
            if self._mask is None:
                self.numpy_mask()
            self._area = np.count_nonzero(self._mask)
        return self._area

    def resize(self, width: int, height: int) -> Mask:
        """Resize the mask.

        Args:
            width (int): Target mask width.
            height (int): Target mask height.

        Returns:
            Mask: A resized Mask.
        """
        if width != self.width() or height != self.height:
            mask = cv2.resize(self.numpy_mask().astype("uint8"), (width, height))
            return Mask(mask=mask.astype("bool"))
        return self
