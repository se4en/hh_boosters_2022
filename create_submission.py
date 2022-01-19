import torch
import os
import sys
from hydra import initialize, compose, initialize_config_dir

from src.predictors.base_predictor import Predictor
from src.utils.utils import get_object, set_seed


if __name__ == "__main__":
    experiment_name = sys.argv[1]

    initialize_config_dir(config_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "outputs/", 
                                                  experiment_name, ".hydra"), job_name="predictor")   
    cfg = compose(config_name="config")

    # init model
    head = get_object(cfg.model.head)
    model = get_object(cfg.model.model, head=head)
    
    # init some params
    batch_size = cfg.training.batch_size
    set_seed(cfg.training.seed)    
    trainer = get_object(cfg.trainer, model=model, batch_size=batch_size)
    test_dataset = get_object(cfg.data.test_data)

    # init predictor
    predictor = Predictor(model=model, trainer=trainer, test_dataset=test_dataset)

    # load weights
    model.load_state_dict(torch.load(os.path.join("outputs/", experiment_name, "model.pth"), 
                                     map_location=trainer.get_device()))
    model.eval()

    preds = predictor.predict()
    preds.to_csv(os.path.join("outputs/", experiment_name, "submission.csv"), index=False, header=True)
