import argparse

from DatasetTools.Config.config import get_cfg
from DatasetTools.Datasets import data_parsers

TOOLS = {

}


def parse_args():

    

    ap = argparse.ArgumentParser()
    
    # Add sub parsers
    parent_parser = ap.add_subparsers(
        title="parser",
        dest="parser",
        required=True
    )
    sub_parsers = []
    for parser in data_parsers.values():
        sub_parsers.append(parser.set_argument_parser(parent_parser))


    

    # for tool in TOOLS:
        # tool.set_parser(sub_parser)
    
    args = ap.parse_args()
    
    parser = data_parsers[args.parser]().parse_args(args)





if __name__ == "__main__":
    parse_args()




# parser = coco_parser.COCODataset()
# parser.parse(
#     "C:/Users/HE7/Desktop/annotations_trainval2017/annotations/instances_val2017.json",
#     "C:/a/b/c"
# )
# images = parser.images()
# print(images[0])
