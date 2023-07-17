import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="defaults", version_base=hydra.__version__)
def main(cfg: DictConfig) -> None:

    data_parser = instantiate(cfg.dataset, cfg=cfg, _recursive_=False)
    data_parser.load()

    if cfg.task._target_ is not None:
        task = instantiate(cfg.task, cfg=cfg, _recursive_=False)
        task.run(data_parser)


if __name__ == "__main__":
    main()
