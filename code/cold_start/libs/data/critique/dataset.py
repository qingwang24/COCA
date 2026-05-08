import os
import random
import torch
import numpy as np
import json
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/code/sft/v5_/lmdb_utils/')
from tools import LmdbReader
def find_last_negative_100_index(lst):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == -100:
            return i
    return -1  # 如果不存在 -100，返回 -1
def remove_consecutive_duplicates(lst, target=151645):
    result = []
    prev = None
    found = False  # 是否发现连续的 151645

    for num in lst:
        if num == target and num == prev:
            found = True  # 发现连续的 151645
            continue  # 跳过当前重复项
        result.append(num)
        prev = num

    return result, found
class Dataset:
    def __init__(self, dataset_meta_path, dataset_pixel_meta_path, seq_length, tokenizer):
        self.dataset_meta_path = dataset_meta_path
        self.dataset_pixel_meta_path = dataset_pixel_meta_path
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.orm_token_num = 5
        self._init()

    def _init(self):
        with open(self.dataset_meta_path, 'r') as f:
            meta_info = json.load(f)
        lmdb_path = meta_info['lmdb_path']
        assert os.path.exists(lmdb_path)
        self.lmdb = LmdbReader(lmdb_path)
        self.total_sentence_len = meta_info['size']
        
    def _fetch_lmdb(self, sample_id):
        return self.lmdb[sample_id]

    def _get_data(self, sample_id):
        data = self._fetch_lmdb(sample_id)
        pixel_values_path = os.path.join('/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/dataset/datasets/mcts_dataset/pixel_values_train', data['pixel_name_string'])
        if not os.path.exists(pixel_values_path):
            return self._get_data(random.randint(0, len(self) - 1))
        pixel_values = torch.load(pixel_values_path).numpy()
        
        input_ids = data['input_ids'].astype(np.int64)[0]
        if len(input_ids) > 2048:
            return self._get_data(random.randint(0, len(self) - 1))
        # score_flag = data['score_flag']
        # if not score_flag:
        labels = data['labels'].astype(np.int64)[0]
        loss_mask = [0 if token==-100 else 1 for token in labels]
        loss_mask = np.array(loss_mask).astype(np.float32)
        
        labels = data['labels'].astype(np.int64)[0]
        attention_mask = np.ones_like(input_ids, dtype=np.int32)
        # else:
        score_loss_mask = np.zeros_like(input_ids, dtype=np.float32)
        if self.orm_token_num > 0:
            index = find_last_negative_100_index(labels)
            last_several_index = list(range(index - 3 - self.orm_token_num, index - 3))
            score_loss_mask[last_several_index] = 1
        score_labels = np.full_like(input_ids, data['score'], dtype=np.float32)
        # labels = score
        # input_ids, flag = remove_consecutive_duplicates(input_ids)
        # if flag:
        #     labels = labels[1:]
        #     loss_mask = loss_mask[1:]
        #     attention_mask = attention_mask[1:]
        #     score_labels = score_labels[1:]
        #     score_loss_mask = score_loss_mask[1:]
        # assert len(labels) == len(input_ids) == len(attention_mask) == len(score_labels) == len(score_loss_mask)
        # assert input_ids.count(151645) == 3
        # assert input_ids[np.where(score_loss_mask == 1)[0][-1]]==151645
        # bool_list = list(map(bool, score_loss_mask))

        return {
            'input_ids':torch.from_numpy(np.array(input_ids)), 'labels':torch.from_numpy(labels),
            'loss_mask':torch.from_numpy(loss_mask), 'attention_mask':torch.from_numpy(attention_mask), 
            'pixel_values':torch.tensor(pixel_values, dtype=torch.float32), 'image_grid_thw':torch.tensor(data['image_grid_thw'][0]),
            'score_labels':torch.from_numpy(score_labels), 'score_loss_mask':torch.from_numpy(score_loss_mask)
            # "score_flag":score_flag
        }


    def __len__(self):
        return self.total_sentence_len

    def __getitem__(self, indices):
        if isinstance(indices, list):
            samples = list()
            for item in indices:
                sample = self._get_data(item)
                samples.append(sample)        
            return samples
        elif isinstance(indices, tuple):
            return self._get_data(indices)
        return self._get_data(indices)




class DataCollator:
    def __init__(self, tokenizer_pad_token_id):
        self.tokenizer_pad_token_id = tokenizer_pad_token_id

    def __call__(self, batch_data):

        def merge1d(tensors, pad_id):
            lengths= [len(s) for s in tensors]
            out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i,:len(s)] = s
            return out

        if isinstance(batch_data, list) and len(batch_data) == 1 and isinstance(batch_data[0], list): # for dataloader(batch_size!=None)
            batch_data = batch_data[0]
        assert len(batch_data) == 1, "暂时只支持batch_size为1"
        input_ids = merge1d([data['input_ids'] for data in batch_data], self.tokenizer_pad_token_id)
        attention_mask = merge1d([data["attention_mask"] for data in batch_data], 0.0)
        labels = merge1d([data['labels'] for data in batch_data], -100)
        loss_mask = merge1d([data["loss_mask"] for data in batch_data], 0.0)
        image_grid_thw = torch.stack([data["image_grid_thw"] for data in batch_data], 0)
        pixel_values = batch_data[0]['pixel_values']
        score_labels = merge1d([data['score_labels'] for data in batch_data], -100)
        score_loss_mask = merge1d([data["score_loss_mask"] for data in batch_data], 0.0)
        # score_flag = batch_data[0]['score_flag']

        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "pixel_values":pixel_values,
            "image_grid_thw":image_grid_thw,
            "labels":labels,
            "loss_mask":loss_mask,
            "score_labels":score_labels,
            "score_loss_mask":score_loss_mask
            # "score_flag":score_flag
        }



if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.append('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/code/prm/v1/libs/data/prm')
    from sampler import SeqSampler
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # Dataset._generate_seq_attention_mask(20, [3, 2, 4])

    data_meta_path = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/prm/v1/dataset/train_debug/prompt.mdb.meta.json'
    data_pixel_meta_path = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/prm/v1/dataset/train_pixel_debug/prompt.mdb.meta.json'
    seq_length = 4096
    tokenizer = AutoTokenizer.from_pretrained("/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-7B-Instruct/")
    sampler = SeqSampler(
        data_meta_path, 
        1, 0, 2, seq_length, epoch=0
    )
    dataset = Dataset(data_meta_path, data_pixel_meta_path, seq_length=seq_length, tokenizer=tokenizer, prm_loss_weight=1.0, prm_select_random_num=-1, prm_select_ret_num=-1, prm_fixed_position=-1, orm_loss_weight=1.0, orm_token_num=5)

    # data_loader = DataLoader(dataset, sampler=sampler, collate_fn=DataCollator(), batch_size=None)
    data_loader = DataLoader(dataset, sampler=sampler, collate_fn=DataCollator(151643), batch_size=1)

    for cur_data in data_loader:
        dev = 233
