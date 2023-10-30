from __future__ import annotations

from pathlib import Path
from typing import Type

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
import random

from DatasetTools.data_parsers.base_parser import BaseParser
from DatasetTools.utils import image_utils
from DatasetTools.utils.utils import path_or_str
from DatasetTools.visualization.draw import draw_image_annotations

from .base_task import BaseTask


class SplitDataset(BaseTask):
    """Split the dataset samples into train, val, test.
    """
    
    # TODO: Test
    def __init__(
        self,
        cfg: DictConfig,
        train_split: float,
        val_split: float,
        test_split: float,
        output_path: path_or_str,
        shuffle: bool,
    ):
        """Create a split-dataset task object.

        Args:
            cfg (DictConfig): A configuration object.
            train_split (float): Percentage of train samples.
            val_split (float): Percentage of validation samples.
            test_split (float): Percentage of test samples.
            output_path (path_or_str): Output dataset path.
            shuffle (bool): Shuffle the samples.
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.output_path = Path(output_path)
        self.shuffle = shuffle

        if output_path is None:
            raise ValueError("output_path can not be None")
        
        if train_split + val_split + test_split > 1:
            raise ValueError(f"Splits must add up to 1 "
                             f"({train_split, val_split, test_split})")

    def run(self, parser: Type[BaseParser]):
        """Run the task.

        Args:
            parser (Type[BaseParser]): A loaded data parser.
        """
        n_samples = len(parser.samples)

        samples = parser.samples
        if self.shuffle:
            samples = samples.copy()
            random.shuffle(samples)

        i_train = round(n_samples * self.train_split)
        i_val = round(n_samples * self.val_split) + i_train
        i_test = round(n_samples * self.test_split) + i_val
        
        train_samples = parser.samples[i_train:]
        val_samples = parser.samples[i_train:i_val]
        test_samples = parser.samples[i_val:i_test]

        print(f"Train samples:\t{len(train_samples)}")
        print(f"Val samples:\t{len(val_samples)}")
        print(f"Test samples:\t{len(test_samples)}")

        if train_samples:
            parser.save(train_samples, self.output_path / "train.json")
        if val_samples:
            parser.save(val_samples, self.output_path / "val.json")
        if test_samples:
            parser.save(test_samples, self.output_path / "test.json")


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
