from typing import Dict, Optional, Any, Tuple, List
import torch
from transformers import Trainer, BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import logging

from src.utils.train_test_split import to_one_hot

logger = logging.getLogger(__name__)


class BaseTrainer(Trainer):
    def __init__(self, model: BertModel, bert_path: str, optimizer = None, train_dataset: Dataset = None, 
                 writer: SummaryWriter = None, val_dataset: Dataset = None, batch_size: int = 16, num_epochs: int = 10, 
                 use_gpu: bool = True, max_len: int = 512, num_workers: int = 0, treshold: float = 0.5, scheduler=None, 
                 merge_feedback: bool = False, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._model = model
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        self._tokenizer = BertTokenizer.from_pretrained(bert_path)
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._writer = writer
        self._treshold = treshold
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._max_len = max_len
        self._use_gpu = use_gpu
        self._num_workers = num_workers
        
        self._true = []
        self._pred = []
        self._best_val_score = 0.0
        self._train_loss = 0.0
        self._train_batch_num = 0

        self._merge_feedback = merge_feedback

        dev = "cpu"
        if self._use_gpu:
            if torch.cuda.is_available():
                dev = "cuda:0"
        self._device = torch.device(dev)
        self._model.to(self._device)

    def get_device(self):
        return self._device

    def _batch_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_dict = self._model(**batch)
        
        if batch["target"] is not None:  # save labels for metrics
            class_probs = output_dict["class_probs"]
            target = batch["target"]
            for i in range(len(target)):
                sample_preds = (class_probs[i, :] > self._treshold).nonzero(as_tuple=True)[0]

                if len(sample_preds) == 0:
                    self._pred.append(list(to_one_hot([torch.argmax(class_probs[i, :]).item()])))
                else:
                    self._pred.append(list(to_one_hot(sample_preds.tolist())))
                self._true.append(target[i, :].tolist())
                
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

    def _encode_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        processed_batch = {"ratings": []}
        if batch[0]["target"] is not None:
            processed_batch["target"] = []

        pos_texts = []
        neg_texts = []
        merged_texts = []

        for sample in batch:
            if self._merge_feedback:
                merged_texts.append(" ".join([sample["positive"], sample["negative"]]))
            else:
                pos_texts.append(sample["positive"])
                neg_texts.append(sample["negative"])
            
            if batch[0]["target"] is not None:
                processed_batch["target"].append(sample["target"])
            processed_batch["ratings"].append([sample["salary_rating"], sample["team_rating"], 
                                               sample["managment_rating"], sample["career_rating"], 
                                               sample["workplace_rating"], sample["rest_recovery_rating"]])

        if self._merge_feedback:
            processed_batch["merged_tokens"], processed_batch["merged_mask"] = self._encode_texts(merged_texts)
        else: 
            processed_batch["pos_tokens"], processed_batch["pos_mask"] = self._encode_texts(pos_texts)
            processed_batch["neg_tokens"], processed_batch["neg_mask"] = self._encode_texts(neg_texts)
   
        processed_batch["ratings"] = torch.LongTensor(processed_batch["ratings"]).to(self._device)
        if batch[0]["target"] is not None:
            processed_batch["target"] = torch.LongTensor(processed_batch["target"]).to(self._device)

        return processed_batch

    def _train_epoch(self, epoch: int) -> None:
        # create new dataloaders for data shuffle
        self._train_dataloader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True,
                                            num_workers=self._num_workers, collate_fn=lambda x: x)
        self._val_dataloader = DataLoader(self._val_dataset, batch_size=self._batch_size, shuffle=True,
                                          num_workers=self._num_workers, collate_fn=lambda x: x)

        # train_losses = []
        self._true = []
        self._pred = []
        self._model.train()

        for batch in self._train_dataloader:
            self._train_batch_num += 1
            self._batch_num += 1

            self._optimizer.zero_grad()

            batch = self._encode_batch(batch)

            loss = self._batch_loss(batch)

            loss.backward()
            loss_value = loss.item()
            self._train_loss += loss_value
            # train_losses.append(loss_value)

            # TODO
            # batch_grad_norm = self._rescale_gradients()

            self._optimizer.step()

            if self._scheduler:
                self._scheduler.step()

            # if len(train_losses) > 10:
            #     train_losses.pop(0)
            self._writer.add_scalar("Loss/train/batch", self._train_loss/self._train_batch_num, self._batch_num)

        self._writer.add_scalar("F1/train/epoch", f1_score(self._true, self._pred, average="samples"), epoch)
        
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                self._writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        val_loss = 0.0
        val_batch_num = 0
        self._true = []
        self._pred = []
        self._model.eval()

        for batch in self._val_dataloader:
            val_batch_num += 1

            batch = self._encode_batch(batch)
            loss = self._batch_loss(batch)
            val_loss += loss.item()

        self._writer.add_scalar("Loss/val/epoch", val_loss/val_batch_num, epoch)
        cur_val_score = f1_score(self._true, self._pred, average="samples")
        self._writer.add_scalar("F1/val/epoch", cur_val_score, epoch)

        if cur_val_score >= self._best_val_score:
            print("Save new best score = {cur_val_score}, epoch = {epoch}")
            self._best_val_score = cur_val_score
            torch.save(self._model.state_dict(), "model.pth")

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
