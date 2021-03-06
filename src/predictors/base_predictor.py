from typing import List, Dict, Tuple, Any

import torch
import torch.nn as nn
from pandas import DataFrame
from transformers import BertTokenizer, Trainer
from torch.utils.data import DataLoader, Dataset


class Predictor:
    def __init__(
        self,
        model: nn.Module,
        trainer: Trainer,
        test_dataset: Dataset,
        treshold: List[float] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ):
        self._model = model
        self._trainer = trainer
        self._test_dataset = test_dataset
        self._test_dataloader = DataLoader(
            self._test_dataset,
            batch_size=self._trainer._batch_size,
            shuffle=False,
            num_workers=self._trainer._num_workers,
            collate_fn=lambda x: x,
        )
        self._treshold = torch.FloatTensor(treshold).to(self._trainer._device)

    def predict_batch(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[List[List[int]], List[List[float]]]:
        result = []
        res_probs = []

        prep_batch = self._trainer._encode_batch(batch)
        output_dict = self._model(**prep_batch)
        class_probs = output_dict["class_probs"]

        for i in range(len(batch)):
            sample_preds = (class_probs[i, :] > self._treshold).nonzero(as_tuple=True)[
                0
            ]
            res_probs.append(class_probs[i, :].tolist())

            # some postprocessing
            if "xa0" in batch[i]:
                if batch[i]["xa0"]:
                    result.append([0])
                    continue

            if len(sample_preds) == 0:
                result.append([torch.argmax(class_probs[i, :]).item()])
                # some postprocessing
                # if class_probs[i, 0] > class_probs[i, 8]:
                #     result.append([0])
                # else:
                #     result.append([8])
            else:
                result.append(sample_preds.tolist())

        # rows, classes = (class_probs > self._treshold).nonzero(as_tuple=True)

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

        return result, res_probs

    def predict(self) -> Tuple[DataFrame, DataFrame]:
        predictions = []
        probabilities = []
        # probabilities = torch.empty((0,9), device="cpu")

        for batch in self._test_dataloader:
            preds, probs = self.predict_batch(batch)

            # probabilities = torch.cat((probabilities, probs.cpu()), dim=0)

            for sample, pred, prob in zip(batch, preds, probs):
                str_pred = (",").join(list(map(lambda x: str(x), pred)))
                predictions.append([sample["review_id"], str_pred])
                probabilities.append([prob, str_pred])

        return DataFrame(predictions, columns=["review_id", "target"]), DataFrame(
            probabilities, columns=["probs", "target"]
        )
