import json
from pathlib import Path

from DatasetTools.structures import bounding_box
from DatasetTools.structures.image import Image
from DatasetTools.structures.instance import Instance
from DatasetTools.structures.mask import Mask
from DatasetTools.utils.utils import get_logger

from .base_parser import BaseParser

logger = get_logger(module_name="COCODataset")


class COCODataset(BaseParser):

    def load(self):
        """Parse a COCO dataset.
        """
        # Get paths
        annotations_file = Path(self.cfg.DATASET.ANNOTATIONS_PATH)
        if self.cfg.DATASET.IMAGES_PATH:
            images_path = Path(self.cfg.DATASET.IMAGES_PATH)
        else:
            images_path = None

        # Load dataset
        data = json.loads(annotations_file.read_text())

        # Save the info into the cfg
        if "info" in data:
            self.cfg.DATASET.META.update(data["info"])

        # Parse categories
        categories = {}
        for cat in data["categories"]:
            if cat["id"] in categories:
                logger.error(f"Duplicated category ID: {cat['id']}")
            categories[cat["id"]] = {
                "name": cat["name"],
                "supercategory": cat["supercategory"]
            }
        self.cfg.DATASET.LABELS = {k: v["name"] for k, v in categories.items()}

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
