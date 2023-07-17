from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from DatasetTools.utils.image_utils import bgr_to_rgb


class matplotlib_visualizer:
    """Image visualizer with matplotlib
    """

    def __init__(self):
        """Create a matplotlib image visualizer.
        """
        pass

    def show(
        self,
        image: np.ndarray,
        image_mode: Literal["RGB", "BGR"] = "RGB"
    ) -> None:
        """Show an image.

        Args:
            image (np.ndarray): The image to show.
            image_mode (Literal["RGB", "BGR"], optional): Image color mode,
                "RGB" or "BGR". Defaults to "RGB".
        """
        if image_mode == "BGR":
            image = bgr_to_rgb(image)
        plt.imshow(image)
        plt.show()
