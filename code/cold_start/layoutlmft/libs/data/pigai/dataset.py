import os
import cv2
import torch
import numpy as np
import json
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path)
from lmdb_utils.utils import LmdbReader


class Dataset:
    def __init__(self, dataset_meta_path, seq_length, tokenizer, prm_loss_weight, prm_select_random_num, prm_select_ret_num, prm_fixed_position, orm_loss_weight, orm_token_num):
        self.dataset_meta_path = dataset_meta_path
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.prm_loss_weight = prm_loss_weight
        self.prm_select_random_num = prm_select_random_num
        self.prm_select_ret_num = prm_select_ret_num
        self.prm_fixed_position = prm_fixed_position
        self.orm_loss_weight = orm_loss_weight
        self.orm_token_num = orm_token_num
        self._init()

    def _init(self):
        with open(self.dataset_meta_path, 'r') as f:
            meta_info = json.load(f)
        lmdb_path = meta_info['lmdb_file']
        assert os.path.exists(lmdb_path)
        self.lmdb = LmdbReader(lmdb_path, sub_dir=False)
        self.total_sentence_len = meta_info['size']

    def _fetch_lmdb(self, sample_id):
        return self.lmdb[sample_id]
    
    @staticmethod
    def _generate_seq_attention_mask(seq_length, sentence_length, dtype=np.float32):
        # step 1. 全赋负无穷
        attention_mask = np.full((seq_length, seq_length), fill_value=-np.inf, dtype=dtype)
        # step 2. 下三角赋0
        attention_mask = np.triu(attention_mask, k=1)
        # step 3. 下三角中, 无关地区赋负无穷
        cum_y, cum_x = 0, 0
        for cur_len in sentence_length:
            cur_start_y = cum_y + cur_len
            cur_start_x = cum_x
            cur_end_x = cum_x + cur_len
            attention_mask[cur_start_y:, cur_start_x:cur_end_x] = -np.inf
            cum_y += cur_len
            cum_x += cur_len
        # step 4. 右下角, 无效区域赋负无穷
        attention_mask[cum_x:, cum_y:] = -np.inf
        return attention_mask

    def _get_seq(self, indices):
        prm_ret_front_tokens = 1 # <ret>前只取一个token(即它自己)
        prm_fixed_position_front_tokens = 1 # 固定位置前只取一个token(即它自己)
        bot_index = 151644 # <|im_start|>
        enter_index = 198 # \n
        tmp_sample = list()
        tmp_prm_weight = list()
        tmp_orm_weight = list()
        for idx, index in enumerate(indices):
            sample_id, offset = index
            elements = self._fetch_lmdb(sample_id)

            if idx != len(indices) - 1:
                new_elements = elements.copy()
                bot_start_index = (new_elements == bot_index).nonzero()[0][-1] + 3 # "<|im_start|>assistant\n"
                answer_len = len(new_elements) - bot_start_index

                cur_prm_weight = np.zeros_like(new_elements, dtype=np.float32)
                cur_orm_weight = np.zeros_like(new_elements, dtype=np.float32)

                # step 1. prm, random
                if self.prm_select_random_num > 0:
                    if answer_len > self.prm_select_random_num + 1:
                        select_index_range = np.arange(bot_start_index, len(new_elements) - 1)
                        random_indices = np.random.choice(select_index_range, size=self.prm_select_random_num, replace=False)
                        cur_prm_weight[random_indices] = self.prm_loss_weight

                # step 2. prm, <ret>
                if self.prm_select_ret_num > 0:
                    is_ret_mask = new_elements == enter_index # <ret>
                    total_is_ret_index = np.where(is_ret_mask)[0]
                    total_is_ret_index = [t for t in total_is_ret_index if t > bot_start_index]
                    if len(total_is_ret_index) > self.prm_select_ret_num + 1:
                        is_ret_index = np.random.choice(total_is_ret_index, size=self.prm_select_ret_num, replace=False)
                        for is_ret_idx in is_ret_index:
                            is_ret_start = max(bot_start_index, is_ret_idx - prm_ret_front_tokens + 1)
                            is_ret_end = is_ret_idx + 1
                            to_set_list = list(range(is_ret_start, is_ret_end))
                            cur_prm_weight[to_set_list] = self.prm_loss_weight
                
                # step 3. prm, fixed position 
                if self.prm_fixed_position > 0:
                    if answer_len > self.prm_fixed_position + 1:
                        segment_ed_index = list(range(bot_start_index + self.prm_fixed_position, len(new_elements) - 1, self.prm_fixed_position))
                        for segment_idx in segment_ed_index:
                            if segment_idx > bot_start_index:
                                segment_start = max(bot_start_index, segment_idx - prm_fixed_position_front_tokens + 1)
                                segment_end = segment_idx + 1
                                to_set_list = list(range(segment_start, segment_end))
                                cur_prm_weight[to_set_list] = self.prm_loss_weight

                # step 4. orm, last several tokens
                if self.orm_token_num > 0:
                    if answer_len > self.orm_token_num + 1:
                        last_several_index = list(range(len(new_elements) - self.orm_token_num, len(new_elements)))
                        cur_orm_weight[last_several_index] = self.orm_loss_weight

                tmp_sample.append(new_elements[index[1]:])
                tmp_prm_weight.append(cur_prm_weight)
                tmp_orm_weight.append(cur_orm_weight)
            else:
                tmp_sample.append([self.tokenizer.pad_token_id] * (index[1]))
                tmp_prm_weight.append([0.0] * (index[1]))
                tmp_orm_weight.append([0.0] * (index[1]))                

        sample = list()
        scores = list()
        prm_weight = list()
        orm_weight = list()
        pad_len = len(tmp_sample[-1])

        sentence_length = []
        for sp in tmp_sample[:-1]:
            real_sp = sp[1:]
            sample.extend(real_sp)
            pad_len += 1
            s = (float(sp[0])) / 1000
            assert s >= 0 and s <= 1
            scores.extend([s] * len(real_sp))
            sentence_length.append(len(real_sp))
        sample.extend([self.tokenizer.pad_token_id] * pad_len)
        scores.extend([0.0] * pad_len)
        assert len(sample) == self.seq_length

        for p_weight in tmp_prm_weight[:-1]:
            real_p_weight = p_weight[1:]
            prm_weight.extend(real_p_weight)
        prm_weight.extend([0.0] * pad_len)
    
        for o_weight in tmp_orm_weight[:-1]:
            real_o_weight = o_weight[1:]
            orm_weight.extend(real_o_weight)
        orm_weight.extend([0.0] * pad_len)

        assert len(sample) == self.seq_length == len(prm_weight) == len(orm_weight) == len(scores)

        loss_mask = np.maximum(prm_weight, orm_weight)
        attention_mask = self._generate_seq_attention_mask(self.seq_length, sentence_length)
         
        # TODO: position ids也改成, 从每个sentence开头0递增

        return {
            'input_ids':sample, 'labels':scores,
            'loss_mask':loss_mask, 'attention_mask':attention_mask
        }

    def _get_data(self, sample_id):
        prm_ret_front_tokens = 1 # <ret>前只取一个token(即它自己)
        prm_fixed_position_front_tokens = 1 # 固定位置前只取一个token(即它自己)
        bot_index = 151644 # <|im_start|>
        enter_index = 198 # \n

        elements = self._fetch_lmdb(sample_id)

        new_elements = elements.copy()
        bot_start_index = (new_elements == bot_index).nonzero()[0][-1] + 3 # "<|im_start|>assistant\n"
        answer_len = len(new_elements) - bot_start_index

        cur_prm_weight = np.zeros_like(new_elements, dtype=np.float32)
        cur_orm_weight = np.zeros_like(new_elements, dtype=np.float32)

        # step 1. prm, random
        if self.prm_select_random_num > 0:
            if answer_len > self.prm_select_random_num + 1:
                select_index_range = np.arange(bot_start_index, len(new_elements) - 1)
                random_indices = np.random.choice(select_index_range, size=self.prm_select_random_num, replace=False)
                cur_prm_weight[random_indices] = self.prm_loss_weight

        # step 2. prm, <ret>
        if self.prm_select_ret_num > 0:
            is_ret_mask = new_elements == enter_index # <ret>
            total_is_ret_index = np.where(is_ret_mask)[0]
            total_is_ret_index = [t for t in total_is_ret_index if t > bot_start_index]
            if len(total_is_ret_index) > self.prm_select_ret_num + 1:
                is_ret_index = np.random.choice(total_is_ret_index, size=self.prm_select_ret_num, replace=False)
                for is_ret_idx in is_ret_index:
                    is_ret_start = max(bot_start_index, is_ret_idx - prm_ret_front_tokens + 1)
                    is_ret_end = is_ret_idx + 1
                    to_set_list = list(range(is_ret_start, is_ret_end))
                    cur_prm_weight[to_set_list] = self.prm_loss_weight
        
        # step 3. prm, fixed position 
        if self.prm_fixed_position > 0:
            if answer_len > self.prm_fixed_position + 1:
                segment_ed_index = list(range(bot_start_index + self.prm_fixed_position, len(new_elements) - 1, self.prm_fixed_position))
                for segment_idx in segment_ed_index:
                    if segment_idx > bot_start_index:
                        segment_start = max(bot_start_index, segment_idx - prm_fixed_position_front_tokens + 1)
                        segment_end = segment_idx + 1
                        to_set_list = list(range(segment_start, segment_end))
                        cur_prm_weight[to_set_list] = self.prm_loss_weight

        # step 4. orm, last several tokens
        if self.orm_token_num > 0:
            if answer_len > self.orm_token_num + 1:
                last_several_index = list(range(len(new_elements) - self.orm_token_num, len(new_elements)))
                cur_orm_weight[last_several_index] = self.orm_loss_weight

        sample = new_elements[1:].astype(np.int64)
        s = (float(new_elements[0])) / 1000
        assert s >= 0 and s <= 1
        score = np.full_like(sample, s, dtype=np.float32)
        attention_mask = np.ones_like(sample, dtype=np.float32)

        loss_mask = np.maximum(cur_prm_weight[1:], cur_orm_weight[1:])
        assert len(sample) == len(loss_mask)
         

        return {
            'input_ids':sample, 'labels':score,
            'loss_mask':loss_mask, 'attention_mask':attention_mask
        }


    def __len__(self):
        return self.total_sentence_len

    def __getitem__(self, indices):
        if isinstance(indices, list):
            samples = list()
            for item in indices:
                sample = self._get_seq(item)
                samples.append(sample)        
            return samples
        elif isinstance(indices, tuple):
            return self._get_seq(indices)
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
        input_ids = merge1d([torch.tensor(data['input_ids'], dtype=torch.long) for data in batch_data], self.tokenizer_pad_token_id)
        attention_mask = merge1d([torch.tensor(data["attention_mask"]) for data in batch_data], 0.0)
        labels = merge1d([torch.tensor(data['labels']) for data in batch_data], -100.0)
        loss_mask = merge1d([torch.tensor(data["loss_mask"]) for data in batch_data], 0.0)
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels":labels,
            "loss_mask":loss_mask,
        }



if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.append('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/code/pigai/v1/libs/data/gma')
    from sampler import SeqSampler
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # Dataset._generate_seq_attention_mask(20, [3, 2, 4])

    data_meta_path = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/pigai/v1/dataset/valid/prompt.mdb.json'
    seq_length = 4096
    tokenizer = AutoTokenizer.from_pretrained("/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-3B-Instruct")
    sampler = SeqSampler(
        data_meta_path, 
        1, 0, 2, seq_length, epoch=0
    )
    dataset = Dataset(data_meta_path, seq_length=seq_length, tokenizer=tokenizer, prm_loss_weight=1.0, prm_select_random_num=5, prm_select_ret_num=5, prm_fixed_position=5, orm_loss_weight=1.0, orm_token_num=5)

    # data_loader = DataLoader(dataset, sampler=sampler, collate_fn=DataCollator(), batch_size=None)
    data_loader = DataLoader(dataset, sampler=sampler, collate_fn=DataCollator(), batch_size=1)

    # Step 6: 使用 DataLoader 迭代数据
    for cur_data in data_loader:
        dev = 233
