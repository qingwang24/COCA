import os
from libs.utils.counter import Counter
from transformers import AutoTokenizer


qwen_checkpoint_path = '/dmx-csy-mix01/cog3/permanent/qkchang/pretrained_models/Qwen2.5-VL-3B-Instruct'
# qwen_checkpoint_path = '/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-VL-3B-Instruct'
qwen_tokenizer = AutoTokenizer.from_pretrained('/dmx-csy-mix01/cog3/permanent/qkchang/pretrained_models/Qwen2.5-VL-7B-Instruct')

# train dataset
train_dataset_meta_path = '/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/dataset/mmc/v6/dataset/train/prompt.mdb.meta.json'
train_pixel_meta_path = ''
train_seq_length = 2048
train_batch_size = 1

# valid dataset
valid_dataset_meta_path = train_dataset_meta_path


# train
orm_token_num = 5


load_only_parameters = True
load_only_parameters_path = ""


# counter for show each item loss
counter = Counter(cache_nums=50)