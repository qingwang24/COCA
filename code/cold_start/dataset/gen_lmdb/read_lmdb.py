import json
import os
import tqdm
import sys
import torch
import numpy as np
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.extend(dir_path)
from lmdb_utils.tools import LmdbReader


if __name__ == "__main__":
    
    data_meta_path = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v1/dataset/train/prompt.mdb.meta.json'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-VL-7B-Instruct/")
    
    with open(data_meta_path, 'r') as f:
        meta_info = json.load(f)
    lmdb_path = meta_info['lmdb_path']
    assert os.path.exists(lmdb_path)
    lmdb = LmdbReader(lmdb_path)
    total_sentence_len = meta_info['size']

    total_length = []
    total_input_ids = []
    id_set = []

    stage1_num = 0
    stage2_num = 0
    stage3_num = 0
    stage4_num = 0
    stage5_num = 0
    for i in tqdm.tqdm(range(total_sentence_len)):
        data = lmdb[i]
        # pixel_values = torch.load(os.path.join('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/datasets/mcts_dataset/pixel_values', data['pixel_name_string'])).numpy()
        # input_ids = data['input_ids'].astype(np.int64)
        score = data['score']
        score_float = score.item()
        if 0 <= score_float and score_float <= 0.2:
            stage1_num += 1
        elif 0.2 < score_float and score_float <= 0.4:
            stage2_num += 1
        elif 0.4 < score_float and score_float <= 0.6:
            stage3_num += 1
        elif 0.6 < score_float and score_float <= 0.8:
            stage4_num += 1
        else:
            stage5_num += 1
        
        if i % 1000 == 0:
            print(f"0~0.2: {stage1_num}, 0.2~0.4: {stage2_num}, 0.4~0.6: {stage3_num}, 0.6~0.8: {stage4_num}, 0.8~1.0: {stage5_num}")
        

        # attention_mask = np.ones_like(input_ids, dtype=np.int32)

        # # for qkchang:
        # input_ids_text = tokenizer.decode(input_ids[0])
        # print(input_ids_text)
        # 需要确保: bon时, prm的输入的token与input_ids保持一致. 
        
