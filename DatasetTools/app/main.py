import argparse

from  omegaconf import OmegaConf

from DatasetTools.Config.config import get_cfg
from DatasetTools.Datasets import data_parsers

TOOLS = {

}


def parse_args():

    cfg = get_cfg()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--cfg",
        help="Path to a YAML configuration file.",
    )
    ap.add_argument(
        "-p",
        "--parser",
        help="Data parser",
        required=True,
        choices=list(data_parsers.keys())
    )
    ap.add_argument(
        "-a",
        "--annotations",
        help="Path to the annotations",
    )
    ap.add_argument(
        "-i",
        "--images",
        help="Path to the images",
    )
    # TODO: Task
    # TODO: --opts

    args = ap.parse_args()
    
    if args.cfg is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.cfg))

    # TODO: add opts to cfg

    if args.annotations is not None:
        cfg.DATASET.ANNOTATIONS_PATH = args.annotations
    
    if args.images is not None:
        cfg.DATASET.IMAGES_PATH = args.images

    parser = data_parsers[args.parser](cfg)
    parser.load()


if __name__ == "__main__":
    parse_args()




# parser = coco_parser.COCODataset()
# parser.parse(
#     "C:/Users/HE7/Desktop/annotations_trainval2017/annotations/instances_val2017.json",
#     "C:/a/b/c"
# )
# images = parser.images()
# print(images[0])
