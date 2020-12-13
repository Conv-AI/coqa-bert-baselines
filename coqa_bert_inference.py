
import os
import time
from transformers import *
from BERTModel import Model
from utils.eval_utils import AverageMeter
from utils.data_utils import prepare_datasets
import torch.nn as nn
import torch
import json
import io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter, defaultdict
import numpy as np
from random import shuffle
import math
import textacy.preprocessing.replace as rep
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_sm')

MODELS = {'BERT': (BertModel,       BertTokenizer,       'bert-base-uncased'),
          'DistilBERT': (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          'RoBERTa': (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          'SpanBERT': (BertModel, BertTokenizer, 'bert-base-cased')}


class ModelLoader:

    def __init__(self, model_name, model_path, device, save_state_dir):
        self.model_name = model_name
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        tokenizer_model = MODELS[model_name]
        self.tokenizer = tokenizer_model[1].from_pretrained(save_state_dir)
        print("Tokenizer loaded")
        #self.tokenizer = tokenizer_model[1](vocab_file=model_path+"/vocab.txt")
        self.model = Model(MODELS[model_name], model_name,
                           model_path, self.device, self.tokenizer)
        restored_params = torch.load(model_path)
        self.model.load_state_dict(restored_params['model'])
        print("Model loaded")

    def preprocess(self, text):
        text = ' '.join(text)
        temp_text = rep.replace_currency_symbols(text, replace_with='_CUR_')
        temp_text = rep.replace_emails(temp_text, replace_with='_EMAIL_')
        temp_text = rep.replace_emojis(temp_text, replace_with='_EMOJI_')
        temp_text = rep.replace_hashtags(temp_text, replace_with='_TAG_')
        temp_text = rep.replace_numbers(temp_text, replace_with='_NUMBER_')
        temp_text = rep.replace_phone_numbers(
            temp_text, replace_with='_PHONE_')
        temp_text = rep.replace_urls(temp_text, replace_with='_URL_')
        temp_text = rep.replace_user_handles(temp_text, replace_with='_USER_')

        doc = nlp(temp_text)
        tokens = []
        for t in doc:
            tokens.append(t.text)
        return tokens

    def prepare_input(self, data,tokenizer, model_name):
        context_list = data["context"].split(" ")
        # question_list = data["question"].split(" ")
        question_list = []
        qlen = len(data["questions"])
        alen = len(data["answers"])
        for i in range(qlen-1):
            q = self.preprocess(data["questions"][i].split(" "))
            a = self.preprocess(data["answers"][i].split(" "))
            question_list.append('<Q{}>'.format(qlen - i - 1))
            question_list.extend(q)
            question_list.append('<A{}>'.format(qlen - i - 1))
            question_list.extend(a)
        question_list.append('<Q>')
        question_list.extend(data["questions"][qlen-1])
        question_length = len(question_list)
        doc_length_available = 512 - question_length - 3
        if model_name == 'RoBERTa':
            doc_length_available = doc_length_available - 3

        paragraph = context_list
        paragraph = self.preprocess(paragraph)
        if model_name != 'RoBERTa' and model_name != 'SpanBERT':
            paragraph = [p.lower() for p in paragraph]
        paragraph_length = len(paragraph)
        start_offset = 0
        doc_spans = []
        while start_offset < paragraph_length:
            length = paragraph_length - start_offset
            if length > doc_length_available:
                length = doc_length_available - 1
                doc_spans.append([start_offset, length, 1])
            else:
                doc_spans.append([start_offset, length, 0])
            if start_offset + length == paragraph_length:
                break
            start_offset += length

        for spans in doc_spans:
            segment_ids = []
            tokens = []
            if model_name == 'RoBERTa':
                tokens.append('<s>')
            for q in question_list:
                segment_ids.append(0)
                if model_name == 'RoBERTa' or model_name == 'SpanBERT':
                    tokens.append(q)
                    #tokenizer.add_tokens([q])
                else:
                    tokens.append(q.lower())
                    #tokenizer.add_tokens([q.lower()])

            if model_name == 'RoBERTa':
                tokens.extend(['</s>', '</s>'])
            else:
                tokens.append('[SEP]')
                segment_ids.append(0)

            #tokenizer.add_tokens(paragraph[spans[0]:spans[0] + spans[1]])
            tokens.extend(paragraph[spans[0]:spans[0] + spans[1]])
            segment_ids.extend([1] * spans[1])
            yes_index = len(tokens)
            tokens.append('yes')
            segment_ids.append(1)
            no_index = len(tokens)
            tokens.append('no')
            segment_ids.append(1)

            if spans[2] == 1:
                tokens.append('<unknown>')
                tokenizer.add_tokens(['<unknown>'])
                segment_ids.append(1)
            if model_name == 'RoBERTa':
                tokens.append('</s>')
            input_mask = [1] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            converted_to_string = tokenizer.convert_ids_to_tokens(input_ids)
            input_ids.extend([0]*(512 - len(tokens)))
            input_mask.extend([0] * (512 - len(tokens)))
            segment_ids.extend([0] * (512 - len(tokens)))

            # start = ex['answer_span'][0]
            # end = ex['answer_span'][1]

            # if start >= spans[0] and end <= spans[1]:
            #     c_known+=1
            #     start = question_length + 1 + start
            #     end = question_length + 1 + end

            # else:
            #     c_unknown+=1
            #     start = len(tokens) - 1
            #     end = len(tokens) - 1
            # if ex['answer'] == 'yes' and tokens[start]!='yes':
            #     start = yes_index
            #     end = yes_index
            # if ex['answer'] == 'no' and tokens[start]!='no':
            #     start = no_index
            #     end = no_index

        inputdata = {'tokens': tokens,
                     # 'answer':tokens[start : end + 1],
                     # 'actual_answer':ex['answer'] ,
                     'input_tokens': input_ids,
                     'input_mask': input_mask,
                     'segment_ids': segment_ids,
                     # 'start':start,
                     # 'end':end
                     }
        return inputdata

    def getResult(self, data):
        inputData = self.prepare_input(data, self.tokenizer, self.model_name)
        output = self.model.forward(inputData)
        print(output)
        return output
