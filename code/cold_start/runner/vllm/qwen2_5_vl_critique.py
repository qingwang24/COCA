import json
import sys
sys.path.append('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA')
from vision_processor import process_vision_info
from tqdm import tqdm  # Import tqdm for the progress bar
import logging
import math
import numpy as np
import re
import pickle
import copy
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import List, Callable, Tuple, Dict, Iterator, Iterable, Union
import transformers
from itertools import takewhile, repeat
from PIL import Image
import re
import requests
import os
import sys 
import json
from tqdm import tqdm
sys.path.append('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA')
# from vision_processor import process_vision_info
import random
import psutil
import argparse
import warnings
import logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
# sys.path.append('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/code/pigai/v1')
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
sys.path.append('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/code/pigai/v1')
from layoutlmft.models.pigai import PigaiModel
# Replace with your local model path

def split_list(lst, n=4):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def expand_list(input_list, times=3):
    return [item for item in input_list for _ in range(times)]

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


def generate_critique(query_list, part_answer_list, wrong_step_list, right_step_list, imgpath_list, model, processor):
    """Generate critique for the erroneous steps"""
    messages_list = []
    text_list = []
    for query, part_answer, wrong_step, right_step, imgpath in zip(query_list, part_answer_list, wrong_step_list, right_step_list, imgpath_list):
        system_msg='You are a knowledgeable teacher who provides clear, concise guidance.'

        user_prompt = """
Please generate a critique based on the following human-machine dialogue to optimize the model's response performance.

1. First, please read the user's input: 
<USER INPUT QUESTION - START> 
<|vision_start|><|image_pad|><|vision_end|>
<USER INPUT QUESTION>
<USER INPUT QUESTION - END>


2. Second, please read the standard answer:
<STANDARD ANSWER PART 1 - START>
<STANDARD ANSWER PART 1>
<STANDARD ANSWER PART 1 - END>
<STANDARD ANSWER PART 2 - START>
<STANDARD ANSWER PART 2>
<STANDARD ANSWER PART 2 - END>

3. After understanding the user's requirements and the standard answer, review the model-generated content: 
<MODEL GENERATED CONTENT PART 1 - START>
<MODEL GENERATED CONTENT PART 1>
<MODEL GENERATED CONTENT PART 1 - END>
<MODEL GENERATED CONTENT PART 2 - START>
<MODEL GENERATED CONTENT PART 2>
<MODEL GENERATED CONTENT PART 2 - END>

3. MODEL GENERATED CONTENT PART 1 is the same as STANDARD ANSWER PART 1 and STANDARD ANSWER PART 2 is of higher quality than MODEL GENERATED CONTENT PART 2. 
Comparing with the standard answer, generate critique comments for the model-generated content based on the following criteria:
    - Correctness: Verify that all numerical data, formulas, and conclusions are correct, and point out any discrepancies like miscalculations or incorrect coefficients. 
    - Logical Clarity: Ensure the reasoning follows a clear, step-by-step structure, and note any abrupt transitions or unclear steps that could confuse the user.
    - Multi-Modal Integration: For content involving images, charts, etc., ensure that the data, labels, and information extracted are accurate, and indicate if any key details or labels are omitted.  
    - Comprehensive Coverage: Check that every part of the question and its sub-points are addressed, and point out any missing elements or insufficient explanations.
    - Mathematical/Logical Soundness: Assess that all derivations and logical steps are rigorously justified, and highlight any leaps in reasoning or skipped intermediate steps.

4. When generating the critique, please note the following:
    - The critique should not mention the comparison process.
    - The critique should not reference the STANDARD ANSWER PART 2.
    - The critique should not include words like 'standard answer', 'STANDARD ANSWER PART 2' or 'STANDARD ANSWER PART 1'.
    - The critique should not reveal the standard answer or correct answer.
    - The critique should not directly provide the final answer.
    - Only point out issues if there is clear redundancy or obvious errors in the model-generated content.
    - Only provide critique for the parts that require modification; do not comment on parts that are correct.
    - The critique should be concise and clear, easy for the model to understand, prioritized by importance, and list a maximum of 5 items.
    - If you are not certain about an issue, do not include it in the critique.
    - If there are no issues with the model-generated content, the critique should simply be “Content is correct, format is correct, no corrections needed.”

Please return the generated critique in the following format:
{
    "critique": ["Modification suggestion", "Modification suggestion", ...]
}


""".replace('<USER INPUT QUESTION>',query).replace('<STANDARD ANSWER PART 1>',part_answer).replace('<MODEL GENERATED CONTENT PART 2>',wrong_step).replace('<STANDARD ANSWER PART 2>',right_step).replace('<MODEL GENERATED CONTENT PART 1>',part_answer).replace('<|im_end|>', '')
        messages = [
            {"role": "user", "content": [{"type":"image","image":imgpath}, {"type": "text", "text": user_prompt}]}
        ]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).replace('user\n<|vision_start|><|image_pad|><|vision_end|>','user\n')
        messages_list.append(messages)
        text_list.append(text)
    images, videos = process_vision_info(messages_list)
    inputs = processor(text=text_list, images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    # generated_ids = model.generate(**inputs, max_new_tokens=1000, top_p=0.9, top_k=50, temperature=0.7)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

import random
def process_jsonl(rank, world_size, test_dataset, output_dir, model, processor, basename, qwen2vl_infer_batch):
    """Main processing flow for the JSONL file"""
    output_path = output_dir+'/'f'{basename}_rank{rank}.json'
    lines = test_dataset[rank::world_size]
    # if os.path.exists(output_path):
    #     ori_id_list = []
    #     with open(output_path, 'r', encoding='utf-8') as infile:
    #         ori_lines = infile.readlines()
    #         for ori_line in ori_lines:
    #             record = json.loads(ori_line)
    #             ori_id_list.append(record['id'])
    #     lines = [line for line in lines if line['id'] not in ori_id_list]

    lines = split_list(lines, qwen2vl_infer_batch)
    records = []
    for record_list in tqdm(lines, desc=f"Processing records_rank_{rank}", unit=f"{qwen2vl_infer_batch}record"):
        question_list = []
        query_list = []
        part_answer_list = []
        wrong_step_list = []
        right_step_list = []
        target_list = []
        imgpath_list = []
        for record in record_list:
            query = extract_prompt_content(record['prompt'])
            part_answer = record['selected_wrong'][0]
            question = record['question']
            query = record['query']
            right_step = (record['selected_right'][0]+record['selected_right'][1])[len(part_answer):]
            wrong_step = record['selected_wrong'][1]
            imgpath = record['imgpath'].replace('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/','/train34/cog8/permanent/bhwei2/pfhu6/shliu19/')
            imgpath = imgpath.replace('shliu19datasets', 'shliu19/datasets')
            target = record['target']
            question_list.append(question)
            query_list.append(query)
            part_answer_list.append(part_answer)
            wrong_step_list.append(wrong_step)
            right_step_list.append(right_step)
            target_list.append(target)
            imgpath_list.append(imgpath)

        critique_list = generate_critique(query_list, part_answer_list, wrong_step_list, right_step_list, imgpath_list, model, processor) 
        with open(output_path, 'a+', encoding='utf-8') as fi:
            for record, critique in zip(record_list, critique_list):
                record['critique'] = critique
                # records.append(record)
                json.dump(record, fi, indent=4, ensure_ascii=False)
                fi.write('\n')



def main(args):
    
    dist.init_process_group(backend="nccl") 
    rank = dist.get_rank()

    if rank==0:
        
        logger.info(args)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    logger.info(args)
    world_size = dist.get_world_size()
    ngpus = torch.cuda.device_count()
    local_rank = rank % ngpus
    logger.info(f"world_size:{world_size}, rank:{rank}, local_rank:{local_rank}")

    # 读取数据集
    test_dataset = json.load(open(args.dataset_path)) # 读取数据集
    basename = os.path.basename(args.dataset_path).split('.')[0]
    output_file = os.path.join(args.output_dir, basename+'.json')
    if os.path.exists(output_file):
        records = json.load(open(output_file))
        ori_id_list = set([r['id'] for r in records])
        test_dataset = [da for da in test_dataset if da['id'] not in ori_id_list]
    logger.info(f'len_test_datasets: {len(test_dataset)}')
    
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels,padding_side='left')
    processor.tokenizer.padding_side='left'
    if 'Qwen2.5-VL' in args.model_path:
        policy_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
        ).to(local_rank)
    else:
        policy_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16,
        ).to(local_rank)
    policy_model.eval()

    basename = os.path.basename(args.dataset_path).split('.')[0]
    process_jsonl(rank, world_size, test_dataset, args.output_dir, policy_model, processor, basename, args.qwen2vl_infer_batch)

    # del policy_model
    # del pigai_model
    # torch.cuda.empty_cache()

    # dist.barrier()  # 这里进行同步，确保所有rank都完成了文件写入
    dist.destroy_process_group()

if __name__ == "__main__":
    # input_file = "/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/dataset/test_2_5vl.json"    # Path to input file
    # output_file = "/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/exp/qwen2.5vl/direct_critique/test_2_5vl.json"  # Path to output file
    
    # process_jsonl(input_file, output_file)
    # print(f"Processing complete! Results have been saved to {output_file}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/exp/debug')
    parser.add_argument("--model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-VL-2B-Instruct') # policy model
    parser.add_argument("--pigai_model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-1.5B-Instruct') # pigai model
    parser.add_argument("--ORM_model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/experiments/prm/v2/train_valid_v1/checkpoint-4407/') # PRM
    parser.add_argument("--dataset_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/dataset/test_2_5vl.json')
    parser.add_argument("--qwen2vl_infer_batch", type=int, default=1)
    parser.add_argument("--best_of_n", type=int, default=1) # BoN
    parser.add_argument("--top_p", type=float, default=0.9) # BoN
    parser.add_argument("--top_k", type=int, default=50) # BoN
    parser.add_argument("--temperature", type=float, default=0.7) # BoN
    parser.add_argument("--max_len", type=int, default=2048) # BoN
    args = parser.parse_args()

    args.max_pixels = 800 * 600
    # my_env = os.environ.copy()  

    # my_env["PATH"] = "/opt/lib/gcc-11.4.0/bin:/opt/lib/cuda-12.1/bin:" + my_env["PATH"] # 防止镜像没有module命令无法load gcc

    # os.environ.update(my_env)
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


    from datetime import datetime
    now = datetime.now()
    current_time_str = now.strftime("%Y%m%d%H")
    exp_name = f'critique_refine_{current_time_str}'
    par_dir = args.output_dir
    # args.output_dir = os.path.join(args.output_dir, exp_name)
    # if os.path.exists(args.output_dir):
    #     dirs = os.listdir(par_dir)
    #     # 统计同名的数量
    #     num = [item for item in dirs if exp_name in item]
    #     num_len = len(num)
    #     args.output_dir = args.output_dir + '_' + str(num_len+1)
    
    main(args)
