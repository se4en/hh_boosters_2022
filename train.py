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

from src.utils.utils import set_seed, get_object

warnings.filterwarnings("ignore")


@hydra.main(config_path="conf", config_name="local-training")
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
    head = get_object(cfg.model.head)
    model = get_object(cfg.model.model, head=head)

    # init datasets
    train_dataset = get_object(cfg.data.train_data)
    val_dataset = get_object(cfg.data.val_data)

    # init optimizer
    optimizer = get_object(cfg.optimizer, params=model.parameters())

    # init scheduler
    scheduler = None
    if "scheduler" in cfg:
        scheduler = get_scheduler(**cfg.scheduler.params, optimizer=optimizer,
                                  num_training_steps=num_epochs * int(len(train_dataset) / batch_size))

    # init trainer
    trainer = get_object(cfg.trainer, model=model, optimizer=optimizer, scheduler=scheduler, writer=writer,
                         num_epochs=num_epochs, batch_size=batch_size, train_dataset=train_dataset,
                         val_dataset=val_dataset)

    try:
        trainer.train()
    finally:
        # save model
        model_name = 'model.pth'
        torch.save(model.state_dict(), model_name)

    writer.close()


if __name__ == '__main__':
    run_model()
