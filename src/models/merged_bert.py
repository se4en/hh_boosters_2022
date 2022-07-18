from typing import List, Dict, Optional, Any

import torch

from src.models.base_bert import BertClassifier, LstmAttention, ClsMlp


class MergedBertClassifier(BertClassifier):
    def forward(
        self,
        merged_tokens: torch.Tensor,
        merged_mask: torch.Tensor,
        ratings: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        merged_outputs = self._bert(
            merged_tokens, token_type_ids=None, attention_mask=merged_mask
        )[0]

        logits = self.head(
            merged_outputs=merged_outputs, merged_mask=merged_mask, ratings=ratings
        )

        result = {"class_probs": torch.sigmoid(logits)}

        if target is not None:
            target = target.type_as(logits)
            result["loss"] = self.loss(logits, target)

        return result


class MergedLstmAttention(LstmAttention):
    def forward(
        self,
        merged_outputs: torch.Tensor,
        merged_mask: torch.Tensor,
        ratings: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_ratings = ratings / 2 - 1.0

        self._pos_hidden = self.init_pos_hidden()

        X_merged = torch.nn.utils.rnn.pack_padded_sequence(
            merged_outputs,
            torch.count_nonzero(merged_mask, dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        X_merged, self._pos_hidden = self._pos_lstm(X_merged, self._pos_hidden)

        # undo the packing operation
        X_merged, _ = torch.nn.utils.rnn.pad_packed_sequence(X_merged, batch_first=True)

        if self._attention:
            merged_dist, merged_outputs = self._pos_attention(
                X_merged, return_attn_distribution=True
            )

        input_features = torch.cat((merged_outputs, norm_ratings), dim=1)
        output_features = self.feedforward(input_features)
        return output_features


class MergedClsMlp(ClsMlp):
    def forward(
        self,
        merged_outputs: torch.Tensor,
        merged_mask: torch.Tensor,
        ratings: torch.Tensor,
    ) -> torch.Tensor:
        norm_ratings = ratings / 2 - 1.0
        input_features = torch.cat(
            (merged_outputs[:, 0, :], norm_ratings), dim=1
        )  # TODO check dim
        output_features = self.feedforward(input_features)
        # class_probs = torch.sigmoid(output_features)
        return output_features
