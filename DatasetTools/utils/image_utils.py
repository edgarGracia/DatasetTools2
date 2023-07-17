from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Tuple[np.ndarray, float, float]:
    """Resize an image to the given width and height.

    Args:
        image (np.ndarray): The image to resize.
        width (Optional[int], optional): Output width.
            If None it will resize the image with the given ``height``
            maintaining the aspect ration. Defaults to None.
        height (Optional[int], optional): Output height.
            If None it will resize the image with the given ``width``
            maintaining the aspect ration. Defaults to None.

    Returns:
        Tuple(np.ndarray, float, float): The resized image, the horizontal
            resize factor and the vertical resize factor.
    """
    h, w, = image.shape[:2]
    if (width is None or width == w) and (height is None or height == h):
        return image, 1, 1
    if width is not None and height is not None:
        return cv2.resize(image, (width, height)), width / w, height / h
    if width is not None:
        f = width / w
    else:
        f = height / h
    return (cv2.resize(image, None, fx=f, fy=f), f, f)


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image.

    Args:
        path (Union[str, Path]): Path to the image file.

    Raises:
        ValueError: If the image cannot be read.

    Returns:
        np.ndarray: Numpy ``BGR`` image of shape (H, W, 3) and ``uint8`` dtype.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image ({path})")
    return img


def write_image(path: Union[str, Path], image: np.ndarray) -> None:
    """Save an image.

    Args:
        path (Union[str, Path]): Output file path.
        image (np.ndarray): A ``BGR`` numpy image of shape (H, W, 3) and
            ``uint8`` dtype.
    """
    cv2.imwrite(str(path), image)
