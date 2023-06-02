# Script for linear evaluation of pre-trained DINO embeddings

import hydra
import pytorch_lightning as pl
import torch
from lightning_fabric.strategies import DDPStrategy
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy

from contextvit.meta_arch import LinearProb


@hydra.main(version_base=None, config_path="../config", config_name="main_linear_prob")
def linear_prob(cfg: DictConfig) -> None:

    # Load the linear probing model (pl-LightningModule)
    model = LinearProb(cfg)

    # set precision to high on A100
    torch.set_float32_matmul_precision("high")

    # Define the trainer.
    # We will use the configurations under cfg.trainer.
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True), **cfg.trainer
    )

    trainer.fit(model=model)


if __name__ == "__main__":
    linear_prob()
