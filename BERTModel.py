import torch
import torch.nn as nn
from utils.eval_utils import compute_eval_metric
from transformers import *
import numpy as np
import collections


class Model(nn.Module):

    def __init__(self, model, model_name, model_path, device, tokenizer):
        super(Model, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.model_name = model_name
        # if model_path is not None:
        #     self.pretrained_model = model[0].from_pretrained(
        #         model_path).to(device)
        # else:
        self.pretrained_model = model[0].from_pretrained(
            model[2]).to(device)
        self.pretrained_model.resize_token_embeddings(len(self.tokenizer))
        self.pretrained_model.train()
        self.qa_outputs = nn.Linear(768, 2).to(device)

    def to_list(self, tensor):
        return tensor.detach().cpu().tolist()

    def forward(self, inputs, train=False):
        inputs = [inputs]
        input_ids = torch.tensor([inp['input_tokens']
                                  for inp in inputs], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([inp['input_mask']
                                   for inp in inputs], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([inp['segment_ids']
                                    for inp in inputs], dtype=torch.long).to(self.device)
        # start_positions = torch.tensor(
        #     [inp['start'] for inp in inputs], dtype=torch.long).to(self.device)
        # end_positions = torch.tensor(
        #     [inp['end'] for inp in inputs], dtype=torch.long).to(self.device)

        if self.model_name == 'BERT' or self.model_name == 'SpanBERT':
            outputs = self.pretrained_model(
                input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        else:
            # DistilBERT and RoBERTa do not use segment_ids
            outputs = self.pretrained_model(
                input_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_logits = self.to_list(start_logits)[0]
        end_logits = self.to_list(end_logits)[0]

        start_idx_and_logit = sorted(
            enumerate(start_logits), key=lambda x: x[1], reverse=True)
        end_idx_and_logit = sorted(
            enumerate(end_logits), key=lambda x: x[1], reverse=True)
        start_indexes = [idx for idx, logit in start_idx_and_logit[:5]]
        end_indexes = [idx for idx, logit in end_idx_and_logit[:5]]

        tokens = self.to_list(input_ids)[0]
        question_indexes = [i+1 for i,
                            token in enumerate(tokens[1:tokens.index(102)])]

        PrelimPrediction = collections.namedtuple(
            "PrelimPrediction", ["start_index",
                                 "end_index", "start_logit", "end_logit"]
        )

        prelim_preds = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                # throw out invalid predictions
                if start_index in question_indexes:
                    continue
                if end_index in question_indexes:
                    continue
                if end_index < start_index:
                    continue
                prelim_preds.append(
                    PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logits[start_index],
                        end_logit=end_logits[end_index]
                    )
                )

        prelim_preds = sorted(prelim_preds, key=lambda x: (
            x.start_logit + x.end_logit), reverse=True)

        BestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "BestPrediction", ["text", "start_logit", "end_logit"]
        )

        nbest = []
        seen_predictions = []
        for pred in prelim_preds:

            # for now we only care about the top 5 best predictions
            if len(nbest) >= 5:
                break

            # loop through predictions according to their start index
            if pred.start_index > 0:  # non-null answers have start_index > 0

                text = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(
                        tokens[pred.start_index:pred.end_index+1]
                    )
                )
                # clean whitespace
                text = text.strip()
                text = " ".join(text.split())

                if text in seen_predictions:
                    continue

                # flag this text as being seen -- if we see it again, don't add it to the nbest list
                seen_predictions.append(text)

                # add this text prediction to a pruned list of the top 5 best predictions
                nbest.append(BestPrediction(
                    text=text, start_logit=pred.start_logit, end_logit=pred.end_logit))

        # and don't forget -- include the null answer!
        nbest.append(BestPrediction(
            text="", start_logit=start_logits[0], end_logit=end_logits[0]))

        #print(nbest)
        return(nbest)
