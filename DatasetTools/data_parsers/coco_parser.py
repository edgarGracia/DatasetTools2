import json
import logging
from pathlib import Path

from DatasetTools.structures import bounding_box
from DatasetTools.structures.image import Image
from DatasetTools.structures.instance import Instance
from DatasetTools.structures.mask import Mask

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class COCODataset(BaseParser):
    """Parse a COCO dataset from a .json file.
    """

    def load(self):
        """Parse and load a COCO dataset.
        """
        # Read dataset annotations
        dataset_path = self.cfg.dataset.annotations_path
        if not dataset_path:
            raise ValueError(f"Invalid dataset path "
                             f"(dataset.annotations_path: '{dataset_path}')")
        with open(dataset_path, "r") as f:
            data = json.load(f)

        # List dataset images
        if self.cfg.dataset.images_path:
            images_path = Path(self.cfg.dataset.images_path)
        else:
            images_path = None

        # Save the info into the cfg
        if "info" in data:
            self.cfg.dataset.meta = data["info"]

        # Parse categories
        categories = {}
        for cat in data["categories"]:
            if cat["id"] in categories:
                logger.error(f"Duplicated category ID: {cat['id']}")
            categories[cat["id"]] = {
                "name": cat["name"],
                "supercategory": cat["supercategory"]
            }
        self.cfg.dataset.labels = {k: v["name"] for k, v in categories.items()}

        # Parse images
        images = {}
        for image_data in data["images"]:
            image_path = image_data["file_name"]
            if images_path is not None:
                image_path = images_path / image_path
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

            images[image_data["id"]] = image

        # Parse annotations
        for annot in data["annotations"]:
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
                label = categories[cat_id]["name"]
            else:
                cat_id = None
                label = None
            if "segmentation" in annot:
                mask = Mask(rle=annot["segmentation"])
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

            if annot["image_id"] in images:
                images[annot["image_id"]].annotations.append(instance)
            else:
                logger.warning(f"Image ID ({annot['image_id']}) not found for "
                               f"annotation: {annot['id']}")

        self._images = list(images.values())

        non_annot = [i for i in self._images if not i.annotations]
        if non_annot:
            logger.warning(f"Found ({len(non_annot)}) images with no "
                           f"annotation: {[i.id for i in non_annot]}")
