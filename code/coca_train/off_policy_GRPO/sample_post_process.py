# import debugpy
# debugpy.connect(('localhost', 5678))
import logging
import math
import numpy as np
import torch
import torch.distributed as dist
import pandas as pd
import os
import sys
import gc
import json
from tqdm import tqdm
from pathlib import Path
import glob
import copy
import argparse
import warnings
import logging
import pickle
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
sys.path.append('./')
sys.path.append('/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/code/sft/v4')
from layoutlmft.models.critique import CritiqueModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from trl.trainer.utils import selective_log_softmax
from trl.extras.profiling import profiling_decorator
from vllm import LLM, SamplingParams
from torch.nn.utils.rnn import pad_sequence
from off_policy_GRPO.utils_new import PigaiAccuracyORM, FormatORM
import torch.nn.functional as F
def _get_per_token_logps(model, inputs):
    from trl.trainer.utils import selective_log_softmax
    logits_to_keep = inputs['logits_to_keep']
    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items() if k not in
        ['logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps', 'rewards_per_func']
    }
    # inputs['pixel_values'] = inputs['pixel_values'].requires_grad_()
    input_ids = inputs['input_ids']
    # with torch.no_grad():
    logits = model(**inputs).logits
    # exclude the last logit: it corresponds to the next token pred
    logits = logits[:, -(logits_to_keep + 1):-1, :]
    logits = logits / 1.0
    input_ids = input_ids[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens
def extract_prompt_content(prompt_str):
    """提取<|vision_end|>和<|im_end|>之间的内容"""
    start_tag = "<|vision_end|>"
    end_tag = "<|im_end|>"
    
    start_idx = prompt_str.find(start_tag)
    if start_idx == -1:
        return None
    
    start_idx += len(start_tag)
    end_idx = prompt_str.find(end_tag, start_idx)
    
    if end_idx == -1:
        return None
    
    return prompt_str[start_idx:end_idx].strip()
def split_list(lst, n=4):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def expand_list(input_list, times=3):
    return [item for item in input_list for _ in range(times)]

def get_chunk(lst, n, k):
    chunk_size = math.ceil(len(lst) / n)  # integer division
    chunks = [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    return chunks[k]







    


    

def json_to_parquet(json_path, output_dir):
    ''''''
    print(json_path)
    print(output_dir)
    save_data_dir = os.path.join(output_dir, 'data')
    save_data_path = os.path.join(save_data_dir, 'train-00000-of-00001.parquet')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_data_dir, exist_ok=True)
    df = pd.read_json(json_path)
    df.to_parquet(save_data_path)
    print(f"成功将 JSON 数据保存为 Parquet 格式到 {save_data_path}")



def main(args):
    dist.init_process_group(backend="nccl") 
    rank = dist.get_rank()

    if rank==0:
        logger.info(args)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(args)
    world_size = dist.get_world_size()
    ngpus = torch.cuda.device_count()
    local_rank = rank % ngpus
    logger.info(f"world_size:{world_size}, rank:{rank}, local_rank:{local_rank}")

    # 读取数据集
    # test_dataset = json.load(open(args.dataset_path)) # 读取数据集


    # 加载policy model, judge model
    # pretrained_path = "/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-7B-Instruct" # hu_debug
    pretrained_path = "/dmx-csy-mix01/cog3/permanent/qkchang/pretrained_models/Qwen2.5-VL-7B-Instruct/"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
    processor = AutoProcessor.from_pretrained(pretrained_path, max_pixels=args.max_pixels)


    critic = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.critic_model_path, torch_dtype=
torch.bfloat16).to(f'cuda:{local_rank}')
    for param in critic.parameters():
        param.requires_grad = False
    critic.eval()


    # actor = Qwen2VLForConditionalGeneration.from_pretrained(args.critic_model_path)
    actor = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.actor_model_path, torch_dtype=
torch.bfloat16).to(f'cuda:{local_rank}')
    for param in actor.parameters():
        param.requires_grad = False
    actor.eval()
    
    exp_name = f'{args.exp_id}'
    # output_dir = os.path.join(os.path.join(args.output_dir, exp_name), 'dataset')
    pkl_dataset_path = os.path.join(args.output_dir, 'pkl')
    pkl_dataset_path_new = os.path.join(args.output_dir, 'pkl_new')
    # pkl_dataset_path = '/train21/cog8/permanent/qkchang/shliu19/critique/rlhf/experiments/iter_v1/sample_output/sample_iter_0/dataset/pkl'
    # pkl_dataset_path_new = '/train21/cog8/permanent/qkchang/shliu19/critique/rlhf/experiments/iter_v1/sample_output/sample_iter_0/dataset/pkl_new'
    if not os.path.exists(pkl_dataset_path_new):
        os.makedirs(pkl_dataset_path_new)
    # pkl_data_path_list = [os.path.join(pkl_dataset_path, item) for item in data_id]
    files = os.listdir(pkl_dataset_path)
    # files = [file for file in files if 'critic' in file][:]
    files = files[rank::world_size]
    for filename in tqdm(files):
        try:
            pkl_data_path_cur = os.path.join(pkl_dataset_path, filename)
            pkl_dataset_path_cur_new = os.path.join(pkl_dataset_path_new, filename)
            if os.path.exists(pkl_dataset_path_cur_new):
                continue
            with open(pkl_data_path_cur, 'rb') as f:
                data = pickle.load(f)
            data_item = data[0]
            data_output = data[1]
            # if data_output['old_per_token_logps'].shape[0] != 1:
            #     continue

            outputs = data_output
            old_per_token_logps = []
            # batch_size = outputs['input_ids'].shape[0]
            # pixel_shape = outputs['pixel_values'].shape[1] # [len, 1176]
            # batch_pixel_values = data_output['pixel_values'].view(batch_size, -1, pixel_shape) # [B, len, pixel_shape]
            cur_pixel_values = torch.load(os.path.join('/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/dataset/datasets/mcts_dataset/pixel_values_train', os.path.basename(data_item['pixel_values_path'])))
            input_ids_list = []
            for b in range(outputs['input_ids'].shape[0]):
                process_slice = slice(b, b+1,)
                cur_inputs = {k: v[process_slice] for k, v in outputs.items() if k not in ['pixel_values', 'logits_to_keep', 'old_per_token_logps', 'ref_per_token_logps', 'advantages', 'rewards_per_func']} # 将inputs分成batch个
                cur_inputs['logits_to_keep'] = outputs['logits_to_keep']

                cur_inputs['pixel_values'] = cur_pixel_values

                # if cur_inputs['input_ids'] not in input_ids_list:
                #     input_ids_list.append(cur_inputs['input_ids'])
                # else:
                #     continue
                if 'critic' in filename:
                    cur_old_per_token_logps = _get_per_token_logps(critic, cur_inputs)
                else:
                    cur_old_per_token_logps = _get_per_token_logps(actor, cur_inputs)
                # cur_old_per_token_logps = _get_per_token_logps(model, cur_inputs) # ref_model_logps [B, completion_length]
                cur_old_per_token_logps = cur_old_per_token_logps.cpu()
                del cur_inputs
                old_per_token_logps.append(cur_old_per_token_logps)
            # if len(input_ids_list) < 4:
            #     continue
            old_per_token_logps = torch.cat(old_per_token_logps, dim=0)
            outputs['old_per_token_logps'] = old_per_token_logps.cpu()
            save_data = [data_item, outputs]
            with open(pkl_dataset_path_cur_new, 'wb') as f:
                pickle.dump(save_data, f)
            # os.remove(pkl_data_path_cur)
        except Exception as e:
            print(f"Error in sample, skip. Erro Info: {e}")
            continue

    dist.destroy_process_group()
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/rlhf/experiments/debug/sample/')
    parser.add_argument("--critic_model_path", type=str, default='/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/code/v4/experiments/train_critique/score_loss_0.1/checkpoint-3500')
    parser.add_argument("--actor_model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-2B-Instruct')
    parser.add_argument("--pigai_model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/experiments/pigai/v1/train_v2/checkpoint-467/') # pigai model
    parser.add_argument("--dataset_path", type=str, default='/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2.5vl/mathv_neg/mathv_neg_part1.json')
    parser.add_argument("--qwen2vl_infer_batch", type=int, default=1)
    parser.add_argument("--num_generation", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--exp_id", type=str, default="sample_iter_0")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4) 
    args = parser.parse_args()
    args.report_to = []
    args.max_pixels = 800 * 600


    rank = int(os.environ['RANK'])
    node_id = rank // 8  # 假设每个节点有 8 张卡
    logging.basicConfig(
        format="Node[{}] %(asctime)s - %(levelname)s - %(message)s".format(rank),
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join('run_log/run-debug-node[{}].log'.format(node_id))),  # os.path.join('train_log', )
            logging.StreamHandler(sys.stdout)
            ]
    )
    logger.setLevel(logging.INFO)

    def handle_exception(exc_type, exc_value, exc_tb):
        # 如果是系统退出异常，忽略
        if exc_type == SystemExit:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        # 记录异常到日志
        logging.error("未捕获的异常", exc_info=(exc_type, exc_value, exc_tb))
    # 设置 sys.excepthook 捕获未处理的异常
    sys.excepthook = handle_exception


    exp_name = f'{args.exp_id}'
    args.output_dir = os.path.join(os.path.join(args.output_dir, exp_name), 'dataset')



    main(args)







