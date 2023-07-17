from __future__ import annotations

from pathlib import Path
from typing import Type

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from DatasetTools.data_parsers.base_parser import BaseParser
from DatasetTools.utils import image_utils
from DatasetTools.utils.utils import path_or_str
from DatasetTools.visualization.draw import draw_image_annotations

from .base_task import BaseTask


class Visualization(BaseTask):
    """Visualize the dataset annotations.
    """

    def __init__(
        self,
        cfg: DictConfig,
        output_path: path_or_str,
        show: bool,
        gui_visualizers: str,
        ext: str = ".png"
    ):
        """Create a visualization task object.

        Args:
            cfg (DictConfig): A configuration object.
            output_path (path_or_str): Output images folder.
            show (bool): Show the images in a GUI.
            gui_visualizers (str): Name of the GUI visualizer.
            ext (str, optional): Image extension. Defaults to ".png".
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.output_path = output_path
        self.show = show
        self.gui_visualizers = gui_visualizers
        self.ext = ext

    def run(self, parser: Type[BaseParser]):
        """Run the visualization task.

        Args:
            parser (Type[BaseParser]): A loaded data parser.
        """
        if self.show:
            visualizer = instantiate(self.gui_visualizers)

        if self.output_path is not None:
            output_path = Path(self.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None

        print("Press ctrl+c to stop")
        try:
            for image in tqdm(parser.images(), unit="img"):
                vis_image = draw_image_annotations(self.cfg, image)
                if output_path is not None:
                    self._write_image(output_path, image.path, vis_image)
                if visualizer:
                    visualizer.show(vis_image, image_mode="BGR")
        except KeyboardInterrupt:
            pass

    def _write_image(
        self,
        output_path: Path,
        input_path: Path,
        image: np.ndarray
    ):
        file_path = output_path / (input_path.stem + self.ext)
        if file_path == input_path:
            raise FileExistsError(
                "Output image path can not be equal to input")
        image_utils.write_image(str(file_path), image)
