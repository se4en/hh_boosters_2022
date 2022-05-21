import os
import sys

import torch
from hydra import initialize, compose, initialize_config_dir
from src.predictors.base_predictor import Predictor
from src.utils.utils import get_object, set_seed
from hydra.utils import instantiate


def submit(experiment_name: str):
    initialize_config_dir(config_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "outputs/", 
                                                  experiment_name, ".hydra"), job_name="predictor")   
    cfg = compose(config_name="config")

    # init model
    model = instantiate(cfg.model)
    
    # init some params
    batch_size = cfg.training.batch_size
    set_seed(cfg.training.seed)    
    trainer = instantiate(cfg.trainer, model=model, batch_size=batch_size)
    test_dataset = instantiate(cfg.data.test_data)

    # init predictor
    predictor = Predictor(model=model, trainer=trainer, test_dataset=test_dataset, 
                          treshold = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # load weights
    model.load_state_dict(torch.load(os.path.join("outputs/", experiment_name, "model.pth"), 
                                     map_location=trainer.get_device()))
    model.eval()

    preds, probs = predictor.predict()
    preds.to_csv(os.path.join("outputs/", experiment_name, "submission.csv"), index=False, header=True)
    probs.to_csv(os.path.join("outputs/", experiment_name, "probabilities.csv"), index=False, header=True)


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    submit(experiment_name)
