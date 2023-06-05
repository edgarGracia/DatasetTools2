import argparse

from  omegaconf import OmegaConf

from DatasetTools.Config.config import get_cfg, update_copy_str
from DatasetTools.Datasets import data_parsers
from DatasetTools.Tasks import tasks



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
    sub_parser = ap.add_subparsers(
        title="task",
        dest="task",
        required=True
    )
    for t in tasks.values():
        t.add_sub_parser(sub_parser)

    args = ap.parse_args()

    if args.cfg is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.cfg))

    cfg.DATASET.PARSER = args.parser

    if args.annotations is not None:
        cfg.DATASET.ANNOTATIONS_PATH = args.annotations
    
    if args.images is not None:
        cfg.DATASET.IMAGES_PATH = args.images

    if args.opts:
        cfg = update_copy_str(cfg, args.opts)

    parser = data_parsers[cfg.DATASET.PARSER](cfg)
    parser.load()

    task = tasks[args.task].from_args(args, cfg)
    task.run(parser)


if __name__ == "__main__":
    parse_args()
