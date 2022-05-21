from typing import List

import pandas as pd
import numpy as np
from skmultilearn.model_selection import IterativeStratification


TRAIN_PATH = "../../data/HeadHunter_train_orig.csv"
TEST_PATH = "../../data/HeadHunter_test.csv"

NEW_TRAIN_PATH = "../../data/HeadHunter_train.csv"
NEW_VAL_PATH = "../../data/HeadHunter_val.csv"


def to_one_hot(labels: List[int]) -> np.ndarray:
    res = np.zeros(9)
    res[labels] = 1
    return res


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    y = train['target']

    y = y.apply(lambda x: list(map(int, x.split(",")))).to_numpy()
    y = np.array([to_one_hot(row) for row in y])

    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 1.0-0.2], random_state=422)
    train_indexes, test_indexes = next(stratifier.split(train, y))

    new_train = train.iloc[train_indexes, :]
    new_val = train.iloc[test_indexes, :]

    new_train.to_csv(NEW_TRAIN_PATH, index=False, header=True)
    new_val.to_csv(NEW_VAL_PATH, index=False, header=True)
