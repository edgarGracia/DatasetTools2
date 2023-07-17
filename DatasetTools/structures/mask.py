from __future__ import annotations

from typing import Optional

import numpy as np
import pycocotools.mask as COCOMask


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

    def numpy_mask(self, save: bool = True) -> np.ndarray:
        """Get the mask as a numpy array.

        Args:
            save (bool, optional): If the mask is stored as an RLE, save the
                converted numpy mask to speed-up future calls. Defaults to True.

        Returns:
            np.ndarray: A numpy mask of shape (H, W) of type ``bool``.
        """
        # TODO: check shape and type!
        if self._mask is None:
            mask = np.asfortranarray(COCOMask.decode(self._rle).astype(bool))
            if save:
                self._mask = mask
            return mask
        return self._mask

    def rle(self, save: bool = True) -> dict:
        """Get the mask an RLE.

        Args:
            save (bool, optional): If the mask is stored as a numpy array,
                save the generated RLE to speed-up future calls.
                Defaults to True.

        Returns:
            dict: The RLE
        """
        # TODO: Check return
        if self._rle is None:
            rle = COCOMask.encode(self._mask)
            if save:
                self._rle = rle
            return rle
        return self._rle

    def height(self) -> int:
        """Get the height of the mask.
        """
        # TODO: Test
        if self._height is not None:
            return self._height
        if self._rle is not None:
            height = self._rle["size"][1]
        else:
            height = self._mask.shape[0]
        self._height = height
        return height

    def width(self) -> int:
        """Get the width of the mask.
        """
        # TODO: Test
        if self._width is not None:
            return self._width
        if self._rle is not None:
            width = self._rle["size"][0]
        else:
            width = self._mask.shape[1]
        self._width = width
        return width

    def area(self) -> int:
        """Get the area of the mask.
        """
        if self._area is not None:
            return self._area
        if self._rle is not None:
            area = COCOMask.area(self._rle)
        else:
            area = np.count_nonzero(self._mask)
        self._area = area
        return area
