import os

import pandas as pd
import numpy as np
from skmultilearn.model_selection import IterativeStratification

from src.utils.utils import to_one_hot


RANDOM_SEED = 422

# round 1

TRAIN_PATH_ORIG = "HeadHunter_train_orig.csv"

TRAIN_PATH = "HeadHunter_train.csv"
VAL_PATH = "HeadHunter_val.csv"

# round 2

TRAIN_PATH_ORIG_NEW = "HeadHunter_new_train.csv"

TRAIN_PATH_NEW = "HeadHunter_train_upd.csv"
VAL_PATH_NEW = "HeadHunter_val_upd.csv"


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    # 1 round
    if os.path.exists(os.path.join(data_dir, TRAIN_PATH_ORIG)):
        train = pd.read_csv(os.path.join(data_dir, TRAIN_PATH_ORIG))

        y = train['target']

        y = y.apply(lambda x: list(map(int, x.split(",")))).to_numpy()
        y = np.array([to_one_hot(row) for row in y])

        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 1.0-0.2], random_state=RANDOM_SEED)
        train_indexes, val_indexes = next(stratifier.split(train, y))

        new_train = train.iloc[train_indexes, :]
        new_val = train.iloc[val_indexes, :]

        new_train.to_csv(os.path.join(data_dir, TRAIN_PATH), index=False, header=True)
        new_val.to_csv(os.path.join(data_dir, VAL_PATH), index=False, header=True)

        # 2 round
        if os.path.exists(os.path.join(data_dir, TRAIN_PATH_ORIG_NEW)):
            train_new = pd.read_csv(os.path.join(data_dir, TRAIN_PATH_ORIG_NEW))
            
            train_full = pd.concat([train, train_new])

            y_full = train_full['target']

            y_full = y_full.apply(lambda x: list(map(int, x.split(",")))).to_numpy()
            y_full = np.array([to_one_hot(row) for row in y_full])

            stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 1.0-0.2], random_state=RANDOM_SEED)
            train_indexes, val_indexes = next(stratifier.split(train_full, y_full))

            new_train = train.iloc[train_indexes, :]
            new_val = train.iloc[val_indexes, :]

            new_train.to_csv(os.path.join(data_dir, TRAIN_PATH_NEW), index=False, header=True)
            new_val.to_csv(os.path.join(data_dir, VAL_PATH_NEW), index=False, header=True)
