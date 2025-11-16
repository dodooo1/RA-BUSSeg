import logging

from .bus import build_bus_semi_loader, build_busloader

logger = logging.getLogger("global")


def get_loader(cfg, seed=0):
    cfg_dataset = cfg["dataset"]

    if cfg_dataset["type"] == "bus_semi":
        train_loader_sup, train_loader_unsup = build_bus_semi_loader(
            "train", cfg, seed=seed
        )
        val_loader = build_busloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader

    elif cfg_dataset["type"] == "bus":
        train_loader_sup = build_busloader("train", cfg, seed=seed)
        val_loader = build_busloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    else:
        raise NotImplementedError(
            "dataset type {} is not supported".format(cfg_dataset)
        )
