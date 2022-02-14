from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Dict
import numpy as np


class BertClassifier(nn.Module):
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
        modules = [self._bert.embeddings, *self._bert.encoder.layer[:num_layers-1]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, pos_tokens: torch.Tensor, pos_mask: torch.Tensor, neg_tokens: torch.Tensor,
                neg_mask: torch.Tensor, ratings: torch.Tensor, target: torch.Tensor = None) -> dict:
        pos_outputs = self._bert(pos_tokens, token_type_ids=None, attention_mask=pos_mask)[0]
        neg_outputs = self._bert(neg_tokens, token_type_ids=None, attention_mask=neg_mask)[0]

        logits = self.head(pos_outputs=pos_outputs, pos_mask=pos_mask, neg_outputs=neg_outputs, 
                           neg_mask=neg_mask, ratings=ratings)

        result = {"class_probs": torch.sigmoid(logits)}

        if target is not None:
            target = target.type_as(logits)
            result["loss"] = self.loss(logits, target)

        return result
    
    def forward(self, merged_tokens: torch.Tensor, merged_mask: torch.Tensor, ratings: torch.Tensor, 
                target: torch.Tensor = None) -> dict:
        merged_outputs = self._bert(merged_tokens, token_type_ids=None, attention_mask=merged_mask)[0]

        logits = self.head(merged_outputs=merged_outputs, merged_mask=merged_mask, ratings=ratings)

        result = {"class_probs": torch.sigmoid(logits)}

        if target is not None:
            target = target.type_as(logits)
            result["loss"] = self.loss(logits, target)

        return result


class LstmAttention(nn.Module):
    def __init__(self, batch_size: int, pos_lstm: nn.Module, neg_lstm: nn.Module = None, num_classes: int = 9,
                 attention: bool = True, mlp: nn.Module = None):
        super().__init__()
        self.batch_size = batch_size
        self._attention = attention
        self._merged_feedback = False

        if neg_lstm is not None:
            self._pos_lstm = pos_lstm
            self._neg_lstm = neg_lstm
        else:
            self._merged_feedback = True
            self._pos_lstm = pos_lstm

        if self._attention:
            self._pos_attention = Attention(2*self._pos_lstm.hidden_size)
            if not self._merged_feedback:
                self._neg_attention = Attention(2*self._neg_lstm.hidden_size) 
        
        dev = "cpu"
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        self._lstm_device = torch.device(dev) 

        if mlp is None:
            self.feedforward = nn.Linear(self._pos_lstm.hidden_size + self._neg_lstm.hidden_size + 6, num_classes)
        else:
            self.feedforward = mlp

    def forward(self, pos_outputs: torch.Tensor, pos_mask: torch.Tensor, neg_outputs: torch.Tensor, 
                neg_mask: torch.Tensor, ratings: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor: 
        norm_ratings = ratings/2 - 1.0
        
        self._pos_hidden = self.init_pos_hidden()
        self._neg_hidden = self.init_neg_hidden()

        X_pos = torch.nn.utils.rnn.pack_padded_sequence(pos_outputs, torch.count_nonzero(pos_mask, dim=1).cpu(), 
                                                        batch_first=True, enforce_sorted=False)
        X_neg = torch.nn.utils.rnn.pack_padded_sequence(neg_outputs, torch.count_nonzero(neg_mask, dim=1).cpu(), 
                                                        batch_first=True, enforce_sorted=False)
        
        X_pos, self._pos_hidden = self._pos_lstm(X_pos, self._pos_hidden)
        X_neg, self._neg_hidden = self._neg_lstm(X_neg, self._neg_hidden)

        # undo the packing operation
        X_pos, _ = torch.nn.utils.rnn.pad_packed_sequence(X_pos, batch_first=True)
        X_neg, _ = torch.nn.utils.rnn.pad_packed_sequence(X_neg, batch_first=True)

        if self._attention:
            pos_dist, pos_outputs = self._pos_attention(X_pos, return_attn_distribution=True)
            neg_dist, neg_outputs = self._neg_attention(X_neg, return_attn_distribution=True)

        input_features = torch.cat((pos_outputs, neg_outputs, norm_ratings), dim=1)
        output_features = self.feedforward(input_features)
        return output_features

    def forward(self, merged_outputs: torch.Tensor, merged_mask: torch.Tensor, ratings: torch.Tensor, 
                target: torch.Tensor = None) -> torch.Tensor:
        norm_ratings = ratings/2 - 1.0
        
        self._pos_hidden = self.init_pos_hidden()

        X_merged = torch.nn.utils.rnn.pack_padded_sequence(merged_outputs, 
                                                           torch.count_nonzero(merged_mask, dim=1).cpu(), 
                                                           batch_first=True, enforce_sorted=False)
        
        X_merged, self._pos_hidden = self._pos_lstm(X_merged, self._pos_hidden)

        # undo the packing operation
        X_merged, _ = torch.nn.utils.rnn.pad_packed_sequence(X_merged, batch_first=True)

        if self._attention:
            merged_dist, merged_outputs = self._pos_attention(X_merged, return_attn_distribution=True)

        input_features = torch.cat((merged_outputs, norm_ratings), dim=1)
        output_features = self.feedforward(input_features)
        return output_features

    def init_pos_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(2*self._pos_lstm.num_layers, self.batch_size, self._pos_lstm.hidden_size)
        hidden_b = torch.randn(2*self._pos_lstm.num_layers, self.batch_size, self._pos_lstm.hidden_size)

        hidden_a = hidden_a.to(self._lstm_device)
        hidden_b = hidden_b.to(self._lstm_device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def init_neg_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(2*self._neg_lstm.num_layers, self.batch_size, self._neg_lstm.hidden_size)
        hidden_b = torch.randn(2*self._neg_lstm.num_layers, self.batch_size, self._neg_lstm.hidden_size)

        hidden_a = hidden_a.to(self._lstm_device)
        hidden_b = hidden_b.to(self._lstm_device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
        

class ClsMlp(nn.Module):
    def __init__(self, in_features: int = 1542, num_classes: int = 9, mlp: nn.Module = None):
        super().__init__()
        if mlp is None:
            self.feedforward = nn.Linear(in_features, num_classes)
        else:
            self.feedforward = mlp

    def forward(self, pos_outputs: torch.Tensor, pos_mask: torch.Tensor, neg_outputs: torch.Tensor, 
                neg_mask: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
        norm_ratings = ratings/2 - 1.0
        input_features = torch.cat((pos_outputs[:, 0, :], neg_outputs[:, 0, :], norm_ratings), dim=1)  # TODO check dim
        output_features = self.feedforward(input_features)
        # class_probs = torch.sigmoid(output_features)
        return output_features

    def forward(self, merged_outputs: torch.Tensor, merged_mask: torch.Tensor, 
                ratings: torch.Tensor) -> torch.Tensor:
        norm_ratings = ratings/2 - 1.0
        input_features = torch.cat((merged_outputs[:, 0, :], norm_ratings), dim=1)  # TODO check dim
        output_features = self.feedforward(input_features)
        # class_probs = torch.sigmoid(output_features)
        return output_features

def new_parameter(*size):
    out = nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out

class Attention(nn.Module):
    """ Simple multiplicative attention"""
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in, reduction_dim=-2, return_attn_distribution=False):
        """
        return_attn_distribution: if True it will also return the original attention distribution
        this reduces the one before last dimension in x_in to a weighted sum of the last dimension
        e.g., x_in.shape == [64, 30, 100] -> output.shape == [64, 100]
        Usage: You have a sentence of shape [batch, sent_len, embedding_dim] and you want to
            represent sentence to a single vector using attention [batch, embedding_dim]
        Here we use it to aggregate the lexicon-aware representation of the sentence
        In two steps we convert [batch, sent_len, num_words_in_category, num_categories] into [batch, num_categories]
        """
        # calculate attn weights
        attn_score = torch.matmul(x_in, self.attention).squeeze()
        # add one dimension at the end and get a distribution out of scores
        attn_distrib = F.softmax(attn_score.squeeze(), dim=-1).unsqueeze(-1)
        scored_x = x_in * attn_distrib
        weighted_sum = torch.sum(scored_x, dim=reduction_dim)
        if return_attn_distribution:
            return attn_distrib.reshape(x_in.shape[0], -1), weighted_sum
        else:
            return weighted_sum
