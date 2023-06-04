from __future__ import annotations

import argparse
from typing import Optional, Type
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

from omegaconf import DictConfig

from DatasetTools.utils.utils import add_opts_arg
from DatasetTools.Config.config import get_cfg
from DatasetTools.Datasets.base_parser import BaseParser
from DatasetTools.Visualization.draw import draw_image_annotations


class Visualization:

    def __init__(
        self,
        show: bool = False,
        show_lib: str = "matplotlib",
        output: Optional[str] = None,
        cfg: Optional[DictConfig] = None
    ):
        self.cfg = get_cfg() if cfg is None else cfg
        self.show = show
        self.show_lib = show_lib
        self.output = Path(output) if output else None

    def run(self, parser: Type[BaseParser]):
        images = parser.images()

        for image in images:
            vis_image = draw_image_annotations(image, self.cfg)
            if self.show:
                self._show(vis_image)
            if self.output is not None:
                # TODO: check input != output
                pass
    
    def _show(self, image: np.ndarray):
        if self.show_lib == "matplotlib":
            self._show_plt(image)
        elif self.show_lib == "cv2":
            self._show_cv2(image)
        else:
            raise NotImplementedError

    def _show_plt(self, image: np.ndarray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.show()

    def _show_cv2(self, image: np.ndarray):
        cv2.imshow(self.__class__.__name__, image)
        cv2.waitKey(0)

    @classmethod
    def add_sub_parser(cls, parent_parser: argparse.ArgumentParser):
        ap = parent_parser.add_parser(
            cls.__name__.lower(),
            help="Visualize the dataset's images and annotations"
        )
        ap.add_argument(
            "--s",
            "--show",
            dest="show",
            action="store_true",
            help="Show the images on a window"
        )
        ap.add_argument(
            "--show-lib",
            choices=["matplotlib", "cv2"],
            default="matplotlib",
            help="Set what library use to show the images"
        )
        ap.add_argument(
            "--o",
            "--output",
            dest="output",
            help="Output path"
        )
        add_opts_arg(ap)

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        cfg: Optional[DictConfig]
    ) -> Visualization:
        return Visualization(
            show=args.show,
            show_lib=args.show_lib,
            output=args.output,
            cfg=cfg
        )
