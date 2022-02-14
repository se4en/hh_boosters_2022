import os
import sys
import hydra
import torch.optim as optim
import torch
import logging
import warnings
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from hydra.utils import instantiate

from src.utils.utils import set_seed, get_object
from create_submission import submit

warnings.filterwarnings("ignore")

config_name = "local-training"
# config_name = "test"
if len(sys.argv) > 1:
    config_name = sys.argv[1]


@hydra.main(config_path="conf", config_name=config_name)
def run_model(cfg: DictConfig) -> None:
    print(cfg)
    run(cfg)


def run(cfg: DictConfig) -> None:
    # init training parameters
    batch_size = cfg.training.batch_size
    num_epochs = cfg.training.num_epochs
    set_seed(cfg.training.seed)

    # init tensorboard
    writer = SummaryWriter(f'runs/{cfg.general.experiment_name}')

    # init model
    model = instantiate(cfg.model)

    # init datasets
    train_dataset = instantiate(cfg.data.train_data)
    val_dataset = instantiate(cfg.data.val_data)

    # init optimizer
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # init scheduler
    scheduler = None
    if "scheduler" in cfg:
        scheduler = get_scheduler(**cfg.scheduler, optimizer=optimizer,
                                  num_training_steps=num_epochs * int(len(train_dataset) / batch_size))

    # init trainer
    trainer = instantiate(cfg.trainer, model=model, optimizer=optimizer, scheduler=scheduler, writer=writer,
                          num_epochs=num_epochs, batch_size=batch_size, train_dataset=train_dataset,
                          val_dataset=val_dataset)

    trainer.train()

    writer.close()


if __name__ == '__main__':
    run_model()
