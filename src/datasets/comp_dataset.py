from heapq import merge
import re
import csv
from dataclasses import asdict
from typing import Dict, Any
from torch.utils.data import Dataset
from transformers import PrinterCallback

from src.utils.feedback_instance import FeedbackInstance
from src.utils.train_test_split import to_one_hot


class CompDataset(Dataset):
    def __init__(self, path: str, training: bool = True, decode_companies: bool = True):
        self._path = path
        self._decode_companies = decode_companies
        self._samples = []
        self._training = training

        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # pass header
            for i, line in enumerate(reader):
                # clean data
                line[0] = int(line[0])
                line[1:5] = list(map(lambda x: x.replace('\xa0', ' '), line[1:5]))
                line[5:11] = list(map(int, line[5:11]))

                if self._decode_companies:
                    line[3] = self._decode_company(line[3])
                    line[4] = self._decode_company(line[4])

                # if not self._training:
                #     line = line[:-1]  # remove target
                if self._training:
                    line[-1] = to_one_hot(list(map(int, line[-1].split(","))))

                self._samples.append(FeedbackInstance(*line))

    @staticmethod
    def _decode_company(text: str) -> str:
        return re.sub(r'([^\*]|^)\*{6}([^\*|$])', r'\1компания\2', text)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self._samples[idx]
        return asdict(sample)


if __name__ == "__main__":
    test = CompDataset("../../data/HeadHunter_val.csv", decode_companies=True)
    print(test[1010])
    print(len(test))
