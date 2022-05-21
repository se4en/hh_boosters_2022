import importlib
import os
import random
from typing import Any, List

import numpy as np
import torch


def load_obj(obj_path: str, default_obj_path: str = '') -> Any:
    """Extract an object from a given path.
    
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py

    Args:
        obj_path (str): Path to an object to be extracted, including the object name.
        default_obj_path (str, optional): Default object path. Defaults to ''.

    Raises:
        AttributeError: When the object does not have the given named attribute.

    Returns:
        Any: Extracted object.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_object(cfg: dict, **kwargs) -> Any:
    """Instantiate object from hydra config. 
    
    Analog of hydra.utils.instantiate, but without using Omegaconf library.
    This is required for loading configs inside docker without Omegaconf library.

    Args:
        cfg (dict): Config with object class name in key '_target_' and constructor arguments.

    Returns:
        Any: Сreated instance.
    """
    _class = load_obj(cfg["_target_"])
    del cfg["_target_"]
    _instance = _class(**cfg, **kwargs)
    return _instance


def to_one_hot(labels: List[int]) -> np.ndarray:
    res = np.zeros(9)
    res[labels] = 1
    return res
