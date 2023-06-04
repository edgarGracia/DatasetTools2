from __future__ import annotations

import argparse
from typing import Optional, Type
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

from omegaconf import DictConfig

from DatasetTools.utils.utils import add_opts_arg
from DatasetTools.Config.config import get_cfg
from DatasetTools.Datasets.base_parser import BaseParser


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
            # TODO: Call visualization method
            # np_img = cv2.imread(str(image.path))
            # plt.imshow(np_img)
            # plt.show()
            if self.show:
                # TODO
                pass
            if self.output is not None:
                # TODO: check input != output
                pass
    
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
