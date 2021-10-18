import os
import logging
import glob
import argparse
import math
import random
from tqdm import tqdm, trange
import pickle
import numpy as np
import json
import rouge_w

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = True
import torch
from torch.utils.data import DataLoader, RandomSampler
from tokenization_unilm import UnilmTokenizer, WhitespaceTokenizer
from modeling_unilm import UnilmForSeq2SeqDecode, UnilmConfig
from transformers import AutoModel, BertTokenizer, AutoConfig, AutoTokenizer
import utils_seq2seq


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = True
MODEL_CLASSES = {
    'unilm': (UnilmConfig, UnilmForSeq2SeqDecode, UnilmTokenizer),
    'albert_1': (AutoConfig, AlbertForSeq2SeqDecode, BertTokenizer)
}

class SummaryModel():
    def __init__(self):
        self.model_name_or_path = '../data/torch_unilm_model'
        self.model_recover_path = '../output_dir_lbh_w/model.9.bin'
        self.max_seq_length = 512
        self.beam_size = 1
        self.max_tgt_length = 128
        self.model_type = 'unilm'
        self.do_lower_case = True
        self.length_penalty = 0

        # 设置gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        config = config_class.from_pretrained(self.model_name_or_path, max_position_embeddings=self.max_seq_length)

        tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path, do_lower_case=self.do_lower_case)

        bi_uni_pipeline.append(utils_seq2seq.Preprocess4Seq2seqDecode(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids,
                                                                  self.max_seq_length, max_tgt_length=self.max_tgt_length))
         
        mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

        model_recover = torch.load(self.model_recover_path)
        self.model = model_class.from_pretrained(self.model_name_or_path, state_dict=model_recover, config=config, mask_word_id=mask_word_id, search_beam_size=self.beam_size, length_penalty=self.length_penalty,
                                            eos_id=eos_word_ids, sos_id=sos_word_id)
                                            # , forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len)
        del model_recover

        self.model.to(device)
        torch.cuda.empty_cache()
        self.model.eval()
        next_i = 0
        
    
    
    def process(self, text):
        src = text
        summary = ""
        return summary

def main():
    summary_model = SummaryModel()
    while True:
        x = input()
        res = summary_model.process(x)
        print(res)

if __name__ == "__main__":
    main()


'''
1. 初始化模型
    a. 实例化模型
    b. 加载模型参数
'''