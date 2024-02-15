#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project ：aste
@IDE     ：PyCharm
@Author  ：Mr. Wireless
@Date    ：2021/12/20 19:28 
@Desc    ：
==================================================
"""
from typing import Text
from torch.utils.data import Dataset
from utils.processor import InputExample, DataProcessor
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
import torch
import random
import json
from collections import Counter


def create_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end+1] = 1
    return mask

class CustomDataset(Dataset):
    """
    An customer class representing txt data reading
    """

    def __init__(self,
                 data_type: "Text",
                 data_dir: "Text",
                 processor: "DataProcessor",
                 tokenizer: "AutoTokenizer",
                 max_seq_length: "int"
                 ) -> "None":
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.sentence_list = []
        if data_type == 'train':
            examples = processor.get_train_examples(data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(data_dir)
        else:
            examples = processor.get_test_examples(data_dir)
        self.examples = examples

    def __getitem__(self, idx: "int"):
        example = self.examples[idx]  # type:InputExample
        #0-1124
        raw_len=len(example.text_a.split(' '))
        inputs = self.tokenizer(example.text_a, max_length=self.max_seq_length, padding='max_length', truncation=True)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        #token_type_ids = inputs.token_type_ids
        spans = example.spans
        relations = example.relations
        span_labels = example.span_labels
        relation_labels = example.relation_labels
        seq_len = len([i for i in input_ids if i != 0])

        token_masks=[1]*raw_len
        context2token_masks = []
        current_token_index = 1
        word2psan={}
        for token in example.text_a.split(' '):
            sub_tokens = self.tokenizer.tokenize(token)
            start_token_index = current_token_index
            end_token_index = current_token_index + len(sub_tokens) - 1
            current_token_index = end_token_index + 1
            for ids in range(start_token_index, end_token_index+1):
                word2psan[ids]=token

        repeat_gt=60
        if repeat_gt != -1:
            if len(spans)!=0:
                k = repeat_gt//len(spans)
                m = repeat_gt%len(spans)
                gt_spans = spans*k + spans[:m]
                gt_types = span_labels*k + span_labels[:m]
                gt_masks = [1]*repeat_gt
                # if not (len(gt_spans) == len(gt_types) == len(gt_masks) == repeat_gt):
                #     print(len(gt_spans),len(gt_types) ,len(gt_masks), repeat_gt)
                assert len(gt_spans) == len(gt_types) == len(gt_masks) == repeat_gt
            else:
                gt_spans=[[0,0]]*repeat_gt
                gt_types=[0]*repeat_gt
                gt_masks = [0]*repeat_gt
            if len(relations)!=0:
                k=repeat_gt//len(relations)
                m=repeat_gt%len(relations)
                gt_relations=relations*k + relations[:m]
                gt_relation_labels=relation_labels*k + relation_labels[:m]
                assert len(gt_relations)==len(gt_relation_labels)==repeat_gt
            else:
                gt_relations=[[0,0,0,0]]*repeat_gt
                gt_relation_labels=[0]*repeat_gt


        return input_ids, attention_mask, spans, relations, span_labels, relation_labels, seq_len,\
              context2token_masks, token_masks, gt_spans, gt_types, gt_masks, raw_len, gt_relations, gt_relation_labels, word2psan

    def __len__(self):
        return len(self.examples)
    

