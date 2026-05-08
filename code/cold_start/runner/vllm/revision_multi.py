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
from vllm import LLM, SamplingParams 
sys.path.append('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/code/pigai/v1')
from layoutlmft.models.pigai import PigaiModel
# Replace with your local model path
def get_pigai_score(question_list, refinement_list, target_list, pigai_model, pigai_processor):
    messages_list = []
    for question, answer, target in zip(question_list, refinement_list, target_list):

        messages = [
            {'role':'system', 'content':'You are a helpful assistant.'},
            {'role':'user', 'content':f"You are playing the role of a teacher. Based on the standard answer, you determine whether the student's response is correct. The question is: {question}, the student's response is: {answer}, and the standard answer is: {target}"},
            {'role':'assistant', 'content':' My grading result is: [score].'},
        ]
        messages_list.append(messages)

    chat_texts = pigai_processor.apply_chat_template(
        messages_list, tokenize=False, add_generation_prompt=False, continue_final_message=False
    )
    for chat_idx in range(len(chat_texts)):
        chat_text = chat_texts[chat_idx]
        if chat_text.endswith('\n'):
            chat_texts[chat_idx] = chat_texts[chat_idx][:-1]
    # tokens = tokenizer(chat_texts).input_ids # tokenizer.batch_decode(tokens, skip_special_tokens=False)[0]
    pigai_tokens = pigai_processor(
                text=chat_texts,
                padding=True,
                return_tensors="pt",
            )
    pigai_tokens = pigai_tokens.to(pigai_model.device)
    with torch.no_grad():
        logits = pigai_model(**pigai_tokens).logits # (batch, seq_len, 1)
        last_token_index = (pigai_tokens['attention_mask'] != 0).sum(-1) - 1
        logits = logits[:, :, 0][list(range(len(logits))), last_token_index]
        score = torch.sigmoid(logits).tolist()
    return score
def process_file(output_dir,file):
        file_path = os.path.join(output_dir, file)
        records=[]
        with open(file_path, 'r', encoding='utf-8') as fi:
            lines = fi.readlines()
            record=''
            flag=False
            for line in lines:
                if record=='' and line=='{\n':                
                    flag = True
                if line=='}\n':
                    flag = False
                record+=line
                if not flag:
                    records.append(json.loads(record)['id'])
                    record = ''
        return records
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


def generate_revision(query_list, original_answer_list, imgpath_list, critique_list, model, processor):
    """Generate critique for the erroneous steps"""
    inputs = []
    for query, original_answer, imgpath, critique in zip(query_list, original_answer_list, imgpath_list, critique_list):
        user_critic='''Please refine the model-generated response by considering the evaluation comments provided below. 
These comments serve as guidance to enhance the accuracy, relevance, and comprehensiveness of the response. 
While incorporating the suggestions, ensure that the final output remains natural and contextually appropriate, rather than strictly adhering to the comments.

User Input:
<|vision_start|><|image_pad|><|vision_end|>
<User Input>

Model Generated Content:
<Model Generated Content>

Corrective Feedback:
<Corrective Feedback>


Based on the evaluation comments, refine the model's response while maintaining natural flow and coherence. 
Use the comments as helpful suggestions rather than strict rules. The final response should be improved but still retain its original style and intent. Output the revised content directly (Note: the output should not include any prefixes or suffixes like <-Start> or <-End>).'''.replace('<User Input>', query).replace('<Model Generated Content>', original_answer.replace('<|im_end|>', '')).replace('<Corrective Feedback>',cri_str)
        messages = [
        {"role": "user", "content": [ {"type": "image", "image": imgpath}, {"type": "text", "text": user_critic}]},

    ]

        # Generate response
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )+f"Let's think step by step."
        text = text.replace('user\n<|vision_start|><|image_pad|><|vision_end|>','user\n')
        images, videos = process_vision_info(messages)
        inputs += [{"prompt":text,"multi_modal_data": {"image": images[0]}}]*10
    samplingparams = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, skip_special_tokens=False)
    generated_ids_trimmed = []
    outputs = model.generate(inputs, samplingparams, use_tqdm=False)
    for out in outputs:
        generated_ids_trimmed.append(out.outputs[0].token_ids)#cumulative_logprob
        
    output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    return output_text

import random
def process_jsonl(rank, world_size, test_dataset, output_dir, model, processor, pigai_model, pigai_processor, basename, qwen2vl_infer_batch):
    """Main processing flow for the JSONL file"""
    output_path = output_dir+'/'f'{basename}_rank{rank}.json'
    lines = test_dataset[rank::world_size]
    lines_list = split_list(lines, qwen2vl_infer_batch)
    for lines in tqdm(lines_list, desc=f"Processing records_rank_{rank}", unit=f"{qwen2vl_infer_batch}record"):
        question_list = []
        query_list = []
        original_answer_list = []
        imgpath_list = []
        target_list =[]
        critique_list = []
        for record in lines:
            # query = extract_prompt_content(record['prompt'])
            question = record['question']
            query = record['query']
            original_answer = record['selected_wrong'][0]+record['selected_wrong'][1]
            imgpath = record['imgpath'].replace('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/', '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/').replace('pfhu6/shliu19datasets/','pfhu6/shliu19/datasets/')
            target = record['target']
            critique_str = record['critique']
            try:
                critique_l = eval(critique_str)['critique']
                critique = ''
                for cri_idx in range(len(critique_l)):
                    critique+=f"""{cri_idx+1}. {critique_l[cri_idx]}\n"""
            except:
                critique=critique_str
            question_list.append(question)
            query_list.append(query)
            original_answer_list.append(original_answer)
            imgpath_list.append(imgpath)
            target_list.append(target)
            critique_list.append(critique)


        revision_list_all = generate_revision(query_list, original_answer_list, imgpath_list, critique_list, model, processor)
        revision_list_all = split_list(revision_list_all, 10)
        for question, target, revision_list, record in zip(question_list, target_list, revision_list_all, lines):
            score_list = get_pigai_score([question]*10, revision_list, [target]*10, pigai_model, pigai_processor)
            with open(output_path, 'a+', encoding='utf-8') as fi:
                record['revision'] = revision_list
                record['pigai_score'] = score_list
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
    files = os.listdir(args.output_dir)
    id_list = []

    # from multiprocessing import Pool, cpu_count
    
    if len(files)>0:
        files = [f for f in files if 'rank' in f]
        if len(files)>0:
            for file in tqdm(files):
                id_list+=process_file(args.output_dir, file)
            id_list = set(id_list)
            test_dataset = [da for da in test_dataset if da['id'] not in id_list]



    random.shuffle(test_dataset)
    logger.info(f'len_test_datasets: {len(test_dataset)}')
    
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels,padding_side='left')
    processor.tokenizer.padding_side='left'
    # if 'Qwen2.5-VL' in args.model_path:
    #     policy_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #         args.model_path, torch_dtype=torch.float16,
    #     ).to(local_rank)
    # else:
    #     policy_model = Qwen2VLForConditionalGeneration.from_pretrained(
    #         args.model_path, torch_dtype=torch.float16,
    #     ).to(local_rank)
    # policy_model.eval()
    policy_model = LLM(model=args.model_path, device=f'cuda:{local_rank}', gpu_memory_utilization=0.8)
    basename = os.path.basename(args.dataset_path).split('.')[0]

    pigai_model = PigaiModel.from_pretrained('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/experiments/pigai/v1/train_v2/checkpoint-467/').to(local_rank)
    pigai_model.eval()
    pigai_processor = AutoProcessor.from_pretrained("/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-1.5B-Instruct")
    process_jsonl(rank, world_size, test_dataset, args.output_dir, policy_model, processor, pigai_model, pigai_processor, basename, args.qwen2vl_infer_batch)

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
    parser.add_argument("--output_dir", type=str, default='/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/exp/debug/revision_multi')
    parser.add_argument("--model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-VL-3B-Instruct') # policy model
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

    my_env = os.environ.copy()  

    my_env["PATH"] = "/home4/intern/ycpan4/miniconda3/envs/mcts_cp39/bin:/opt/lib/gcc-11.4.0/bin:/opt/lib/cuda-12.1/bin:" + my_env["PATH"] # 防止镜像没有module命令无法load gcc
    os.environ['CUDA_HOME'] = '/opt/lib/cuda-12.1'
    os.environ.update(my_env)
    os.environ['TRITON_CACHE_DIR'] = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/.triton/autotune'
    os.environ['CXX'] = 'g++'
    os.environ["WANDB_DISABLED"] = 'true'
    rank = int(os.environ['RANK'])
    node_id = rank // 8  # 假设每个节点有 8 张卡
    logging.basicConfig(
        format="Node[{}] %(asctime)s - %(levelname)s - %(message)s".format(rank),
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            # logging.FileHandler(os.path.join('run_log/run-debug-node[{}].log'.format(node_id))),  # os.path.join('train_log', )
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

    
    main(args)
