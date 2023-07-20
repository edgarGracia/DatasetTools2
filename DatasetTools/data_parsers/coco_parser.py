import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig

from DatasetTools.structures import bounding_box
from DatasetTools.structures.image import Image
from DatasetTools.structures.instance import Instance
from DatasetTools.structures.mask import Mask
from DatasetTools.structures.sample import Sample
from DatasetTools.utils.utils import path_or_str

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class COCODataset(BaseParser):
    """Parse a COCO dataset from a JSON file.
    """

    def __init__(
        self,
        cfg: DictConfig,
        annotations_path: path_or_str,
        images_path: Optional[path_or_str] = None,
        image_list: Optional[Union[path_or_str, List[path_or_str]]] = None
    ):
        """Create a COCO dataset parser.

        Args:
            cfg (DictConfig): A configuration object.
            annotations_path (path_or_str): Dataset JSON file.
            images_path (Optional[path_or_str], optional): Dataset images path.
            image_list (Union[path_or_str, List[path_or_str]], optional):
                A list or a file with a list of image or annotation file names
                to process.
        """
        self.cfg = cfg
        self.images_path = Path(images_path)
        self.annotations_path = Path(annotations_path)
        
        self._samples: List[Sample] = []
        self._meta: Dict[str, any] = {}
        self._categories: Dict[int, str] = {}
        self._super_categories: Dict[int, str] = {}

        if image_list is not None:
            if isinstance(image_list, (Path, str)):
                self._only_image_list = [
                    Path(i.strip()).stem
                    for i in Path(image_list).read_text().splitlines()
                ]
            else:
                self._only_image_list = [Path(i).stem for i in image_list]
        else:
            self._only_image_list = None

    def load(self):
        """Parse and load a COCO dataset.
        """
        # Read dataset annotations
        try:
            with open(self.annotations_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.exception(f"Error reading the annotations file "
                             f"{self.annotations_path}")

        # Save the info and licenses data
        if "info" in data:
            self._meta["info"] = data["info"]
        if "licenses" in data:
            self._meta["licenses"] = data["licenses"]

        # Parse categories
        for cat in data["categories"]:
            if cat["id"] in self._categories:
                logger.error(f"Duplicated category ID: {cat['id']}")
            self._categories[cat["id"]] = cat["name"]
            self._super_categories[cat["id"]] = cat["supercategory"]

        # Parse images
        images = {}
        discard_img_id = set()
        for image_data in data["images"]:
            image_path = image_data["file_name"]
            if self.images_path is not None:
                image_path = self.images_path / image_path
            meta = {
                k: image_data[k] for k in set(image_data.keys()).difference({
                    "width", "height", "id", "file_name"
                })
            }
            image = Image(
                path=image_path,
                id=image_data["id"],
                width=image_data.get("width", None),
                height=image_data.get("height", None),
                meta=meta if meta else None
            )
            if image_data["id"] in images:
                logger.error(f"Duplicated image ID: {image_data['id']}")
            if (self._only_image_list is None or
                image.path.stem in self._only_image_list):
                images[image_data["id"]] = image
            else:
                discard_img_id.add(image_data["id"])

        # Parse annotations
        annotations = {}
        for annot in data["annotations"]:
            image_id = annot["image_id"]
            if image_id in discard_img_id:
                continue
            if "bbox" in annot:
                box = bounding_box.BoundingBoxX1Y1WH(
                    annot["bbox"][0],
                    annot["bbox"][1],
                    annot["bbox"][2],
                    annot["bbox"][3],
                    coords_type=bounding_box.CoordinatesType.ABSOLUTE
                )
            else:
                box = None
            if "category_id" in annot:
                cat_id = annot["category_id"]
                label = self._categories[cat_id]
            else:
                cat_id = None
                label = None
            if "segmentation" in annot:
                mask = Mask(
                    rle=annot["segmentation"],
                    width=images.get(image_id, {}).width,
                    height=images.get(image_id, {}).height
                )
            else:
                mask = None

            extras = {
                k: annot[k] for k in set(annot.keys()).difference({
                    "segmentation", "area", "bbox", "category_id", "id"
                })
            }

            instance = Instance(
                bounding_box=box,
                label=label,
                label_id=cat_id,
                id=annot.get("id", None),
                mask=mask,
                extras=extras
            )

            annotations.setdefault(image_id, []).append(instance)
            
        # Create samples
        for image_id in sorted(list(annotations.keys())):
            annots = annotations[image_id]
            if image_id not in images:
                logger.warning(f"Image ID ({image_id}) not found for "
                               f"annotations: {[i.id for i in annots]}")
            sample = Sample(
                image=images.get(image_id, None),
                annotations=annots
            )
            self._samples.append(sample)

        non_annot = set(images.keys()) - set(annotations.keys())
        if non_annot:
            logger.warning(f"Found ({len(non_annot)}) images with no "
                           f"annotation: {non_annot}")

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def samples(self) -> List[Sample]:
        return self._samples
    
    @property
    def labels(self) -> dict:
        return self._categories
