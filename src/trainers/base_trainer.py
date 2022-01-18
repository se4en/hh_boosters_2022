from typing import Dict, Optional, Any, Tuple, List
import torch
from transformers import Trainer, BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)


class BaseTrainer(Trainer):
    def __init__(self, model: BertModel, bert_path: str, optimizer, train_dataset: Dataset, writer: SummaryWriter,
                 val_dataset: Dataset, batch_size: int = 16, num_epochs: int = 10, use_gpu: bool = True,
                 max_len: int = 512, num_workers: int = 0, scheduler=None, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._model = model
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        self._tokenizer = BertTokenizer.from_pretrained(bert_path)
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._writer = writer
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._max_len = max_len
        self._use_gpu = use_gpu
        self._num_workers = num_workers

        dev = "cpu"
        if self._use_gpu:
            if torch.cuda.is_available():
                dev = "cuda:0"
        self._device = torch.device(dev)
        self._model.to(self._device)

    def _batch_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_dict = self._model(**batch)
        loss = output_dict["loss"]
        return loss

    def _encode_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = []

        for text in texts:
            cur_input_ids = self._tokenizer.encode(text=text,
                                                   add_special_tokens=True,
                                                   max_length=self._max_len,
                                                   truncation=True,
                                                   padding=False)
            input_ids.append(cur_input_ids)

        max_len = max([len(sample) for sample in input_ids])

        padded_ids = []
        attn_masks = []

        for sample in input_ids:
            num_pads = max_len - len(sample)
            padded_input = sample + [self._tokenizer.pad_token_id] * num_pads
            attn_mask = [1] * len(sample) + [0] * num_pads

            padded_ids.append(padded_input)
            attn_masks.append(attn_mask)

        tokens = torch.LongTensor(padded_ids).to(self._device)
        mask = torch.LongTensor(attn_masks).to(self._device)

        return tokens, mask

    def _encode_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        processed_batch = {}
        processed_batch["pos_tokens"], processed_batch["pos_mask"] = self._encode_texts(batch["positive"])
        processed_batch["neg_tokens"], processed_batch["neg_mask"] = self._encode_texts(batch["negative"])

        print(batch["salary_rating"])
        # batch["team_rating"]
        # batch["managment_rating"]
        # batch["workplace_rating"]
        # batch["rest_recovery_rating"]

        processed_batch["ratings"] = 0  # TODO

        if "target" in batch:
            processed_batch["target"] = processed_batch["target"].to(self._device)

        return batch

    def _train_epoch(self, epoch: int) -> None:
        # create new dataloaders for data shuffle
        self._train_dataloader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True,
                                            num_workers=self._num_workers, collate_fn=lambda x: x)
        self._val_dataloader = DataLoader(self._val_dataset, batch_size=self._batch_size, shuffle=True,
                                          num_workers=self._num_workers, collate_fn=lambda x: x)

        train_loss = 0.0
        train_batch_num = 0
        self._model.train()

        for batch in self._train_dataloader:
            train_batch_num += 1
            self._batch_num += 1

            self._optimizer.zero_grad()

            batch = self._encode_batch(batch)
            loss = self._batch_loss(batch)

            loss.backward()
            train_loss += loss.item()

            # TODO
            # batch_grad_norm = self._rescale_gradients()

            self._optimizer.step()

            if self._scheduler:
                self._scheduler.step()

            self._writer.add_scalar("Loss/train/batch", train_loss/train_batch_num, self._batch_num)

        for name, param in self._model.named_parameters():
            self._writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        val_loss = 0.0
        val_batch_num = 0
        self._model.eval()

        for batch in self._val_dataloader:
            val_batch_num += 1

            batch = self._encode_batch(batch)
            loss = self._batch_loss(batch)
            val_loss += loss.item()

        self._writer.add_scalar("Loss/val/epoch", val_loss/val_batch_num, epoch)

    # def _rescale_gradients(self) -> Optional[float]:
    #     if self._grad_norm:
    #         parameters_to_clip = [p for p in self._model.parameters()
    #                               if p.grad is not None]
    #         return sparse_clip_norm(parameters_to_clip, self._grad_norm)
    #     return None

    def train(self):
        self._batch_num = 0
        for epoch in range(self._num_epochs):
            self._train_epoch(epoch)
