import os
from libs.utils.counter import Counter
from transformers import AutoTokenizer


qwen_checkpoint_path = '/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-1.5B-Instruct'
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_checkpoint_path)

# train dataset
train_dataset_meta_path = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/pigai/v1/dataset/train/prompt.mdb.json'
train_seq_length = 1024
train_batch_size = 4

# valid dataset
valid_dataset_meta_path = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/pigai/v1/dataset/valid/prompt.mdb.json'
# valid_dataset_meta_path = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/pigai/v1/dataset/small_valid/prompt.mdb.json'


# train
orm_token_num = 5


load_only_parameters = True
load_only_parameters_path = ""


# counter for show each item loss
counter = Counter(cache_nums=1000)