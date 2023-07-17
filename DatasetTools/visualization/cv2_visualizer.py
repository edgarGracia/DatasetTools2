from typing import Literal

import cv2
import numpy as np

from DatasetTools.utils.image_utils import rgb_to_bgr


class cv2_visualizer:
    """Image visualizer with opencv
    """

    def __init__(self):
        """Create a opencv image visualizer.
        """
        pass

    def show(
        self,
        image: np.ndarray,
        image_mode: Literal["RGB", "BGR"] = "RGB",
        window_name: str = "Visualization"
    ) -> None:
        """Show an image.

        Args:
            image (np.ndarray): The image to show.
            image_mode (Literal["RGB", "BGR"], optional): Image color mode,
                "RGB" or "BGR". Defaults to "RGB".
            window_name (str, optional): Window name. Defaults to
                "Visualization".
        """
        if image_mode == "RGB":
            image = rgb_to_bgr(image)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
