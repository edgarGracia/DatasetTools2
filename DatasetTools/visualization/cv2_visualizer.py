from typing import Literal, Optional

import cv2
import numpy as np

from DatasetTools.utils.image_utils import rgb_to_bgr, resize_image


class cv2_visualizer:
    """Image visualizer with opencv
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        window_name: str = "Visualization"
    ):
        """Create a opencv image visualizer.

        Args:
            width (Optional[int], optional): Image width. Defaults to None.
            height (Optional[int], optional): Image height. Defaults to None.
            window_name (str, optional): Window name. Defaults to "Visualization".
        """
        self.width = width
        self.height = height
        self.window_name = window_name

    def show(
        self,
        image: np.ndarray,
        image_mode: Literal["RGB", "BGR"] = "RGB",
    ) -> None:
        """Show an image.

        Args:
            image (np.ndarray): The image to show.
            image_mode (Literal["RGB", "BGR"], optional): Image color mode,
                "RGB" or "BGR". Defaults to "RGB".
        """
        image, _, _ = resize_image(image, self.width, self.height)
        if image_mode == "RGB":
            image = rgb_to_bgr(image)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(0)
