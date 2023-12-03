#-*-coding:utf-8-*-
# Created by Microsoft Corporation
# Licensed under the MIT license.

# Modified by Adaptive Intelligence Research Lab(https://air.changwon.ac.kr/)., 2020. 01. ~

import os
import json
import logging
import torch
import sys
import re
import numpy as np
from torch.utils import data
from time import time


class Dataset(data.Dataset):
    def __init__(self, file_path, tokenizer, 
                 max_source_length=None, 
                 max_target_length=None,
                 large_dataset=False):
        
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.large_dataset = large_dataset

        if max_source_length is None: self.max_source_length = 10e+10
        else: self.max_source_length = max_source_length
        if max_target_length is None: self.max_target_length = 10e+10
        else: self.max_target_length = max_target_length

        if self.large_dataset:
            self.set_file_list()
            data_max_source, data_max_target = self.load_file(self.file_list.pop())
            self.data = list()
        else:
            assert not os.path.isdir(self.file_path)
            data_max_source, data_max_target = self.load_file(self.file_path)

            self.data_size = len(self.data)
        
        if max_source_length is None: self.max_source_length = data_max_source
        if max_target_length is None: self.max_target_length = data_max_target

        logging.info('Total batch : {}'.format(self.data_size))
        logging.info('Max Source Length : {}'.format(self.max_source_length))
        logging.info('Max Target Length : {}'.format(self.max_target_length))
    
    def __getitem__(self, index):
        if self.large_dataset:
            if len(self.data) == 0:
                if len(self.file_list) == 0: self.set_file_list()
                filename = self.file_list.pop()
                self.load_file(filename)
            return self.data.pop(0)
        else:
            return self.data[index]

    def __len__(self):
        ## for setting total_batch, 10%: removed data if source_length < 5
        #return int(1000000 * len(self.file_list) * 0.9)
        return self.data_size

    def set_file_list(self):
        if os.path.isdir(self.file_path):
            file_list = sum([[os.path.join(d[0],f) for f in d[-1]] for d in list(os.walk(self.file_path))],[])
            file_list = sorted(file_list)
        else:
            file_list = [self.file_path]
        self.file_list = file_list
        self.data_size = self.set_data_size(file_list)
    
    def set_data_size(self, file_list):
        data_size = 0
        for filename in self.file_list:
            if os.path.splitext(filename)[-1] in ['.json']:
                with open(filename, 'r') as f: data = json.load(f)
                data_size += len(data)
            else: ## .txt, .tsv, .jsonl
                with open(filename, 'r') as f:
                    for line in f: data_size += 1
        return data_size

    def load_file(self, filename):
        logging.info('Load {} '.format(filename))
        
        sources = list()
        source_lengths = list()
        targets = list()
        target_lengths = list()

        with open(filename, 'r') as ifp:
            key = os.path.splitext(filename)[-1]
            if key in ['.jsonl']:
                data = [json.loads(d) for d in ifp]
            elif key in ['.json']:
                data = json.load(ifp)
            elif key in ['.tsv','.txt']:
                data = [d.strip().split('\t') for d in ifp]
                data = [{'source':d[1], 'target':d[2], '_id':d[0]} for d in data]
            else:
                raise KeyError('No rule for {} filetype.'.format(key))

        self.data = list()
        for index, item in enumerate(data):
            sys.stderr.write('\r{}/{}'.format(index, len(data)))
            start_time = time()
            source = item['sentence']
            target = item['tag']
            if type(source) == list: source = source[0]
            if type(target) == list: target = target[0]
            source = self.cleaning(source)
            target = self.cleaning(target)

            source_length = len(self.tokenizer.tokenize(source))
            target_length = len(self.tokenizer.tokenize(target))

            source_lengths.append(source_length)
            target_lengths.append(target_length)
            
            self.data.append({
                'source':source,
                'target':target,
                'source_length':source_length,
                'target_length':target_length,
                'data':item,
                })
        sys.stderr.write('\n'); sys.stderr.flush()

        import pandas as pd
        logging.info(f"\n{pd.Series(source_lengths).describe()}")
        logging.info(f"\n{pd.Series(target_lengths).describe()}")
        
        return max(source_lengths+[0]), max(target_lengths+[0]) 

    def cleaning(self, sentence):

        sent = sentence.strip()

        #sent = re.sub('\[[^\]]*\]','',sent) ## 대괄호 제거
        #sent = re.sub('\([^\)]*\)','',sent) ## 소괄호 제거
        #sent = re.sub('[^ㅏ-ㅣㄱ-ㅎ가-힣0-9a-zA-Z\.%, ]',' ', sent) ## 특수문자 모두 제거
        sent = re.sub('  *',' ',sent).strip() ## 다중 공백 제거

        return sent

    def convert_sentence_to_input(self, inputs, max_len, direction='right', special_token=False):
        inputs = self.tokenizer.tokenize(inputs)
        if special_token: inputs = [self.tokenizer.cls_token] + inputs + [self.tokenizer.sep_token] ## for bert

        dif = abs(max_len - len(inputs))
        if direction == 'left':
            if len(inputs) < max_len: inputs = ( [self.tokenizer.pad_token]*dif ) + inputs
            elif max_len < len(inputs): inputs = inputs[dif:]
        else:
            if len(inputs) < max_len: inputs += [self.tokenizer.pad_token] * dif
            elif max_len < len(inputs): inputs = inputs[:max_len]
        inputs = self.tokenizer.convert_tokens_to_ids(inputs)
        return inputs
    
    def convert_tokens_to_input(self, inputs, max_len, direction='right', special_token=False):
        if special_token: inputs = [self.tokenizer.cls_token] + inputs + [self.tokenizer.sep_token] ## for bert
        dif = abs(max_len - len(inputs))
        if direction == 'left':
            if len(inputs) < max_len:  inputs = ( [self.tokenizer.pad_token] * dif ) + inputs
            elif max_len < len(inputs):  inputs = inputs[dif:]
        else:
            if len(inputs) < max_len:  inputs += [self.tokenizer.pad_token] * dif
            elif max_len < len(inputs):  inputs = inputs[:max_len]

        inputs = self.tokenizer.convert_tokens_to_ids(inputs)
        return inputs

    def generator_collate_fn(self, data, tokenizer, max_source_length, max_target_length):

        result = {
                'source': list(),
                'target': list(),
                'source_attention_mask': list(),
                'target_attention_mask': list(),
                'target_length': list(),
                'data':list()
                }

        for items in data:
            # source
            source = items['source']
            tokenized_source = tokenizer.encode(source, max_length=max_source_length, padding='max_length', truncation=True)
            #attention_mask = tokenizer.get_special_tokens_mask(tokenized_source, already_has_special_tokens=True)
            enc_attention_mask = [0 if d == tokenizer.pad_token_id else 1 for d in tokenized_source]

            # target
            target = items['target']
            tokenized_target = tokenizer.encode(target, max_length=max_target_length, padding='max_length', truncation=True)
            dec_attention_mask = [0 if d == tokenizer.pad_token_id else 1 for d in tokenized_target]

            target_length = tokenized_target.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in tokenized_target else len(tokenized_target)

            result['data'].append(items['data'])

            result['source'].append(tokenized_source)
            result['target'].append(tokenized_target)
            result['source_attention_mask'].append(enc_attention_mask)
            result['target_attention_mask'].append(dec_attention_mask)
            result['target_length'].append(target_length)
        
        for key in [d for d in result if d not in ['data']]:
            result[key] = torch.tensor(result[key])

        return result 

    def classifier_collate_fn(self, data, tokenizer, max_source_length, max_target_length, labels=None):

        result = {
                'source': list(),
                'target': list(),
                'source_attention_mask': list(),
                'data':list()
                }

        for items in data:
            # source
            source = items['source']
            tokenized_source = self.convert_sentence_to_input(source, max_source_length)
#             tokenized_source = tokenizer.encode(source, max_length=max_source_length, padding='max_length', truncation=True)
            enc_attention_mask = [0 if d == tokenizer.pad_token_id else 1 for d in tokenized_source]

            # target
            target = items['target']
            tokenized_target = labels.index(target)

            result['data'].append(items['data'])

            result['source'].append(tokenized_source)
            result['target'].append(tokenized_target)
            result['source_attention_mask'].append(enc_attention_mask)

        for key in [d for d in result if d not in ['data']]:
            result[key] = torch.tensor(result[key])

        return result 
