from transformers import BertModel
import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np


class DPBertClassifier(nn.Module):
    def __init__(self, bert_path: str, head: nn.Module, bert_layers_to_freeze: int = 12):
        super().__init__()
        self._bert = BertModel.from_pretrained(bert_path)
        self._bert_layers_to_freeze = bert_layers_to_freeze
        self._freeze_bert_layers(self._bert_layers_to_freeze)

        self.head = head
        self.loss = torch.nn.BCEWithLogitsLoss()

    def get_trainable_params(self) -> int:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        trainable_params = sum([np.prod(p.size()) for p in model_parameters])
        return trainable_params

    def _freeze_bert_layers(self, num_layers: int = 12) -> None:
        if num_layers == 0:
            return
        modules = [self.bert.embeddings, *self.bert.encoder.layer[:num_layers-1]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, pos_tokens: torch.Tensor, pos_mask: torch.Tensor, neg_tokens: torch.Tensor,
                neg_mask: torch.Tensor, ratings: torch.Tensor, target: torch.Tensor = None) -> dict:
        pos_outputs = self.bert(pos_tokens, token_type_ids=None, attention_mask=pos_mask)[0]
        neg_outputs = self.bert(neg_tokens, token_type_ids=None, attention_mask=neg_mask)[0]

        logits = self.head(pos_outputs, neg_outputs, ratings)

        result = {"class_probs": torch.sigmoid(logits)}

        if target is not None:
            target = target.type_as(logits)
            result["loss"] = self.loss(logits, target)

        return result


# class LstmAttention(nn.Module):
#     def __init__(self, in_features: int, num_classes: int = 9):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=768,
#                            hidden_size=hidden_size,
#                            num_layers=self.lstm_layers,
#                            dropout=dropout,
#                            bidirectional=bidirectional,
#                            batch_first=False)
#         self.feedforward = nn.Linear(in_features, num_classes)
#
#     def forward(self, pos_outputs: torch.Tensor, neg_outputs: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
#         norm_ratings = ratings/5 - 0.5
#         input_features = torch.cat((pos_outputs, neg_outputs, norm_ratings), dim=1)
#         output_features = self.feedforward(input_features)
#         class_probs = torch.sigmoid(output_features)
#
#         X_batch, x_lens = inputs
#
#         # print("X_batch", X_batch)
#
#         self.hidden = self.init_hidden()
#
#         if torch.isnan(X_batch).any():
#             print("X before pad contains NAN")
#             print("X=", X_batch)
#
#         X = torch.nn.utils.rnn.pack_padded_sequence(X_batch, x_lens, batch_first=True, enforce_sorted=False)
#
#         # if torch.isnan(X).any():
#         #     print("X after pad contains NAN")
#
#         # print("packed X", X)
#
#         # now run through LSTM
#         X, self.hidden = self.rnn(X, self.hidden)
#
#         # if torch.isnan(X).any():
#         #     print("X after lstm contains NAN")
#
#         # print("X before UNPAD", X)
#
#         # undo the packing operation
#         X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
#
#         if torch.isnan(X).any():
#             print("X after unpad contains NAN")
#
#
#         cur_logits = self.classifier_feedforward(X[0, :x_lens[0], :])
#         class_log_probs = F.log_softmax(cur_logits, dim=1)
#
#
#         return class_probs
#
#     def init_hidden(self):
#         # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
#         hidden_a = torch.randn(2*self.lstm_layers, self.batch_size, self.hidden_size)
#         hidden_b = torch.randn(2*self.lstm_layers, self.batch_size, self.hidden_size)
#
#         hidden_a = hidden_a.to(self.device)
#         hidden_b = hidden_b.to(self.device)
#
#         hidden_a = Variable(hidden_a)
#         hidden_b = Variable(hidden_b)
#
#         return (hidden_a, hidden_b)


class ClsMlp(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 9):
        super().__init__()
        self.feedforward = nn.Linear(in_features, num_classes)

    def forward(self, pos_outputs: torch.Tensor, neg_outputs: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
        norm_ratings = ratings/5 - 0.5
        input_features = torch.cat((pos_outputs[:, 0, :], neg_outputs[:, 0, :], norm_ratings), dim=1)  # TODO check dim
        output_features = self.feedforward(input_features)
        # class_probs = torch.sigmoid(output_features)
        return output_features
