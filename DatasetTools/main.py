import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from DatasetTools.data_parsers import create_parser


@hydra.main(config_path="conf", config_name="defaults", version_base=hydra.__version__)
def main(cfg: DictConfig) -> None:

    data_parser = create_parser(cfg)
    data_parser.load()

    if cfg.task._target_ is not None:
        task = instantiate(cfg.task, cfg=cfg, _recursive_=False)
        task.run(data_parser)


if __name__ == "__main__":
    main()
