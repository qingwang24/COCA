from transformers.utils import logging
import os
import json
import numpy as np
import time

logger = logging.get_logger(__name__)


class SeqSampler:
    def __init__(self, dataset_meta_path, world_size, rank_id, batch_size, seq_length, epoch=0):
        self.dataset_meta_path = dataset_meta_path
        self.world_size = world_size
        self.rank_id = rank_id
        self.batch_size = batch_size
        self.seq_length=seq_length
        self.random_state = np.random.RandomState(20250215)
        self.epoch = epoch
        self._init_dataset()
        self._generate_cur_batches()

    def _init_dataset(self):
        with open(self.dataset_meta_path, 'r') as f:
            meta_info = json.load(f)
        self.total_sentence_len = meta_info['size']

    def _generate_cur_batches(self):
        total_valid_index = list(range(self.total_sentence_len))
        self.random_state.shuffle(total_valid_index)
        mini_batches = []
        cur_batch = []
        for sample_index in total_valid_index:
            if len(cur_batch) < self.batch_size:
                cur_batch.append(sample_index)
            else:
                mini_batches.append(cur_batch)
                cur_batch = [sample_index]
        self._cur_batches = mini_batches

    def __iter__(self):
        self.random_state = np.random.RandomState(int(time.time()))
        self.random_state.shuffle(self._cur_batches)
        align_batch_num = (len(self._cur_batches) // self.world_size) * self.world_size
        valid_batches = self._cur_batches[:align_batch_num]
        for batch_idx in range(self.rank_id, len(valid_batches), self.world_size):
            yield valid_batches[batch_idx]

    def __len__(self):
        return len(self._cur_batches) // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch




if __name__ == "__main__":
    sampler = SeqSampler(
        '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/prm/v1/dataset/train/prompt.mdb.meta.json', 
        1, 0, 2, 4096, epoch=0
    )
    output = []
    for idx in sampler:
        output.append(idx)
    dev = 233
