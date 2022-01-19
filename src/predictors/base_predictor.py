import os
from unittest.result import failfast
from wsgiref.headers import Headers
import torch
import torch.nn as nn
from pandas import DataFrame
from hydra import initialize, compose, initialize_config_dir
from transformers import BertTokenizer, Trainer
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Any

from src.utils.utils import get_object 
from src.utils.feedback_instance import FeedbackInstance


class Predictor:
    def __init__(self, model: nn.Module, trainer: Trainer, test_dataset: Dataset, treshold: float = 0.5):
        self._model = model
        self._trainer = trainer
        self._test_dataset = test_dataset
        self._test_dataloader = DataLoader(self._test_dataset, batch_size=self._trainer._batch_size, shuffle=False,
                                           num_workers=self._trainer._num_workers, collate_fn=lambda x: x)
        self._treshold = treshold

    def predict_batch(self, batch: List[Dict[str, Any]]) -> List[List[int]]:
        result = []
        
        prep_batch = self._trainer._encode_batch(batch)
        output_dict = self._model(**prep_batch)
        class_probs = output_dict["class_probs"]

        # print("class_probs", output_dict["class_probs"])

        for i in range(len(batch)):
            sample_preds = (class_probs[i, :] > self._treshold).nonzero(as_tuple=True)[0]

            if len(sample_preds) == 0:
                result.append([torch.argmax(class_probs[i, :]).item()])
            else:
                result.append(sample_preds.tolist())

        # rows, classes = (class_probs > self._treshold).nonzero(as_tuple=True)
        
        # print("\nRows", rows)
        # print("Classes", classes)

        # prev_row = rows[0]
        # buff = [classes[0].item()]
        # for cur_row, cur_class in zip(rows[1:], classes[1:]):
        #     if cur_row == prev_row:
        #         buff.append(cur_class.item())  # add another label
        #     else: 
        #         result.append(buff)  # finish previous sample
        #         prev_row = cur_row
        #         buff = []

        #         if cur_row > prev_row + 1:  # were rows without >.5 probs
        #             for zero_row in range(prev_row + 1, cur_row):
        #                 best_class = torch.argmax(class_probs[zero_row, :]).item()
        #                 result.append([best_class])

        #         buff.append(cur_class.item())  # add first label for current sample
        # result.append(buff) 

        return result

    def predict(self) -> DataFrame:
        predictions = []
        
        for batch in self._test_dataloader:
            preds = self.predict_batch(batch)

            if len(batch) != len(preds):
                print(len(preds))

            for sample, pred in zip(batch, preds):
                str_pred = (",").join(list(map(lambda x: str(x), pred)))
                predictions.append([sample["review_id"], str_pred])
            
        return DataFrame(predictions, columns=["review_id", "target"])
