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
my_env = os.environ.copy()  

my_env["PATH"] = "/home4/intern/ycpan4/miniconda3/envs/mcts_cp39/bin:/opt/lib/gcc-11.4.0/bin:/opt/lib/cuda-12.1/bin:" + my_env["PATH"] # 防止镜像没有module命令无法load gcc
os.environ['CUDA_HOME'] = '/opt/lib/cuda-12.1'
os.environ.update(my_env)
os.environ['TRITON_CACHE_DIR'] = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/.triton/autotune'
os.environ['CXX'] = 'g++'
os.environ["WANDB_DISABLED"] = 'true'
import sys 
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
sys.path.append('/train34/cog8/permanent/bhwei2/pfhu6/shliu19')
# from vision_processor import process_vision_info
import random
import psutil
import argparse
import warnings
import logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
sys.path.append('/train21/cog8/permanent/qkchang/shliu19/critique/code/v4')
from layoutlmft.models.critique import CritiqueModel
# sys.path.append('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/code/pigai/v1')
from layoutlmft.models.pigai import PigaiModel


# Replace with your local model path
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

def get_pigai_score(question, answer, target, pigai_model, pigai_processor):
    messages = [[
        {'role':'system', 'content':'You are a helpful assistant.'},
        {'role':'user', 'content':f"You are playing the role of a teacher. Based on the standard answer, you determine whether the student's response is correct. The question is: {question}, the student's response is: {answer}, and the standard answer is: {target}"},
        {'role':'assistant', 'content':' My grading result is: [score].'},
    ]]

    chat_texts = pigai_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, continue_final_message=False
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
    return score[0]

def Qwen2_VL_inference(query_list, imgpath_list, model, processor, args):
    messages_list = []
    text_list = []
    inputs = []
    for query, imgpath in zip(query_list, imgpath_list):
        messages = [{"role": "user", 
                    "content": [
                        {"type": "image", "image": imgpath}, 
                        {"type": "text", "text": query}]
                    }]
                    # messages = [{"role": "user", "content": [{"type": "image", "image": imgpath}]}]
        text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )+'''Let's think step by step.'''
        images, videos = process_vision_info(messages)
        inputs += [{"prompt":text,"multi_modal_data": {"image": images[0]}}]
    # generated_ids = model.generate(**inputs, max_new_tokens=512,top_p=args.top_p,top_k=args.top_k,temperature=args.temperature)
    # samplingparams = SamplingParams(max_tokens=2048, temperature=0, top_k=1, skip_special_tokens=False)
    samplingparams = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=1,
                    repetition_penalty=1.05,
                    max_tokens=2048,
                    stop_token_ids=[],
                    skip_special_tokens=False,
                )
    # samplingparams = SamplingParams(max_tokens=2048, top_p=0.9,top_k=50,temperature=0.7, skip_special_tokens=False)
    generated_ids_trimmed = []
    with torch.no_grad():
        outputs = model.generate(inputs, samplingparams, use_tqdm=False)
        for out in outputs:
            generated_ids_trimmed.append(out.outputs[0].token_ids)#cumulative_logprob
            
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
    return output_text


def generate_critique(query_list, original_answer_list, imgpath_list, model, processor):
    """Generate critique for the erroneous steps"""
    messages_list = []
    text_list = []
    for query, original_answer, imgpath in zip(query_list, original_answer_list, imgpath_list):
        system_msg='You are a knowledgeable teacher who provides clear, concise guidance.'

        user_prompt = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n
        Please evaluate the model’s answer to the user’s input and generate critique for improvement. When generating the critique, please note:
            - The critique should be concise and clear, easy for the model to understand.
            - If the model’s answer is correct, the critique should simply be: "No corrections needed."

        Now, review the user's input and model's answer:
        User's Input:
        <|vision_start|><|image_pad|><|vision_end|>
        <User Input>

        Model's Answer:
        <Model Generated Content><|im_end|>\n<|im_start|>assistant\n'''.replace('<User Input>',query).replace('<Model Generated Content>',original_answer.replace('<|im_end|>', ''))
        messages = [
            {"role": "user", "content": [{"type":"image","image":imgpath}, {"type": "text", "text": user_prompt}]}
        ]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).replace('user\n<|vision_start|><|image_pad|><|vision_end|>','user\n')
        messages_list.append(messages)
        text = text.replace('user\n<|vision_start|><|image_pad|><|vision_end|>','user\n')
        text_list.append(text)
    images, videos = process_vision_info(messages_list)
    inputs = processor(text=[text_list[0].split('\n<|im_start|>assistant\n')[0]], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model(**inputs, return_dict=True)
        logits = output['score_logits'] # [1, len, 1]
        last_token_index = (inputs['attention_mask'] != 0).sum(-1) - 1
        logits = logits[:, :, 0][list(range(len(logits))), last_token_index]
        score = torch.sigmoid(logits).tolist()
        inputs = processor(text=text_list, images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
        # generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = model.generate(**inputs, max_new_tokens=2048, top_p=0.9, top_k=50, temperature=0.7)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    return output_text, score
def generate_revision(query_list, original_answer_list, imgpath_list, critique_list, model, processor, args):
    """Generate critique for the erroneous steps"""
    messages_list = []
    text_list = []
    inputs=[]
    for query, original_answer, imgpath, critique in zip(query_list, original_answer_list, imgpath_list, critique_list):
        try:
            critique = eval(critique)
            cri_score = critique['score']
            critique_l = critique['critique']
            cri_str = ''
            for cri_id in range(len(critique_l)):
                cri_str+=f"""{cri_id+1}. {critique_l[cri_id]}\n"""
        except:
            cri_str = critique.split('''"score"''')[0].split('{')[0]
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
        inputs += [{"prompt":text,"multi_modal_data": {"image": images[0]}}]
    # samplingparams = SamplingParams(max_tokens=2048, temperature=0, skip_special_tokens=False)
    # samplingparams = SamplingParams(max_tokens=2048, top_p=0.9,top_k=50,temperature=0.7, skip_special_tokens=False)
    samplingparams = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.05,
                    max_tokens=2048,
                    stop_token_ids=[],
                    skip_special_tokens=False,
                )
    # samplingparams = SamplingParams(max_tokens=2048, top_p=0.9,top_k=50,temperature=0.7, skip_special_tokens=False)
    generated_ids_trimmed = []
    with torch.no_grad():
        outputs = model.generate(inputs, samplingparams, use_tqdm=False)
        for out in outputs:
            generated_ids_trimmed.append(out.outputs[0].token_ids)#cumulative_logprob
            
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
    return output_text
import random
def process_jsonl(rank, world_size, test_dataset, output_dir, model, processor, critique_model, critique_processor, pigai_model, pigai_processor, basename, args):
    """Main processing flow for the JSONL file"""
    output_path = output_dir+'/'f'{basename}_rank{rank}.json'
    lines = test_dataset[rank::world_size]
    records = []
    for record in tqdm(lines, desc="Processing records", unit="record"):
        prompt = record['prompt']
        query = extract_prompt_content(prompt)
        question = query.split('Question:')[-1]
        # question = record['question']
        # query = record['query']
        imgpath = record['imgpath'].replace('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/', '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/').replace('pfhu6/shliu19datasets/','pfhu6/shliu19/datasets/')
        target = record['target']

        choice_dict = {i - 65: chr(i) for i in range(65, 91)}
        choice_dict_reverse = {chr(i): i - 65 for i in range(65, 91)}
        if 'mathvista' in record['id'].lower():
            if 'choices' in record.keys():
                choices = record['choices']
                if choices is not None and choices !=[]:
                    if target in choices:
                        target_index = choices.index(target)
                        target = choice_dict[target_index]
                        record['target'] = target
        choice_target = None
        if 'choice_target' in record.keys():
            choice_target=record['choice_target']
        else:
            if 'choices' in record.keys():
                choices = record['choices']
                if choices is not None and choices !=[]:
                    choice_target = choices[int(choice_dict_reverse[target])]



        original_answer = '''Let's think step by step.''' + Qwen2_VL_inference([query], [imgpath], model, processor, args)[0]
        # original_answer = Qwen2_VL_inference([query], [imgpath], model, processor, args)[0]
        # original_answer = record['predict'].split('assistant\n')[-1].replace('<|im_end|>','')
        record['original_answer'] = original_answer
        current_answer = original_answer
        record['revision'] = []
        record['critique'] = []
        record['score_all'] = []
        record['infer_score_all'] = []
        record['score_all_choice'] = []
        score = get_pigai_score(question, original_answer, target, pigai_model, pigai_processor)
        if choice_target is not None:
            score_ = get_pigai_score(question, original_answer, choice_target, pigai_model, pigai_processor)
            record['score_all_choice'].append(score_)
            # score = max(score, score_)

        record['score_all'].append(score)
        for play_idx in range(args.max_play_num):
            current_critique, infer_score = generate_critique([query], [current_answer], [imgpath], critique_model, critique_processor)
            current_critique = current_critique[0]
            if type(infer_score)==list:
                infer_score=infer_score[0]

            try:
                critique_l = eval(current_critique)['critique']
                critique = ''
                for cri_idx in range(len(critique_l)):
                    critique+=f"""{cri_idx+1}. {critique_l[cri_idx]}\n"""
            except:
                critique=current_critique.split("score")[0]
            record['infer_score_all'].append(infer_score)
            record['critique'].append(current_critique)
            current_revision = generate_revision([query], [current_answer], [imgpath], [critique], model, processor, args)
            current_revision = """Let's think step by step.""" + current_revision[0]
            record['revision'].append(current_revision)
            current_answer = current_revision
            score = get_pigai_score(question, current_answer, target, pigai_model, pigai_processor)
            if choice_target is not None:
                score_ = get_pigai_score(question, current_answer, choice_target, pigai_model, pigai_processor)
                record['score_all_choice'].append(score_)
                # score = max(score, score_)
            record['score_all'].append(score)
        current_critique, infer_score = generate_critique([query], [current_answer], [imgpath], critique_model, critique_processor)
        current_critique = current_critique[0]
        if type(infer_score)==list:
            infer_score=infer_score[0]
        record['infer_score_all'].append(infer_score)
        record['critique'].append(current_critique)
        # final_answer = record['revision'][-1]                   
        # score = get_pigai_score(question, final_answer, target, pigai_model, pigai_processor)
        record['pigai_score'] = record['score_all'][0]
        with open(output_path, 'a+', encoding='utf-8') as fi:
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
    dataset_list = ['MathVision_mini',
                    'MathVision',
                    'MathVista_mini',
                    'mmmu_val',
                    'mmmu_val_choice',
                    'test_aug', # charQA
                    'test_human', # charQA
                    'val_aug', # charQA
                    'val_human', # charQA
                    'train_human',
                    'train_aug',
                    'M3COT_test',
                    'M3COT_val',
                    'M3COT_test_choice',
                    'M3COT_val_choice',
                    'MathV360k']
    test_dataset1 = []
    mmmu_id_list = set([da['id'].lower() for da in test_dataset if 'mmmu_val' in da['id'].lower()])
    for da in test_dataset:
        id_name = da['id'].lower()
        if 'mathvista' not in id_name: #and 'mathverse' not in id_name:
            continue
        if 'mmmu_val' in id_name and 'choice' not in id_name:
            if id_name+'_choice' in mmmu_id_list:
                continue
        if 'val_aug' in id_name or 'val_human' in id_name:
            continue
        if 'm3cot' in id_name and ('test' not in id_name or 'choice' not in id_name):
            continue
        test_dataset1.append(da)
    test_dataset = test_dataset1
    basename = os.path.basename(args.dataset_path).split('.')[0]
    output_file = os.path.join(args.output_dir, basename+'.json')
    if os.path.exists(output_file):
        records = json.load(open(output_file))
        ori_id_list = set([r['id'] for r in records])
        test_dataset = [da for da in test_dataset if da['id'] not in ori_id_list]
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
    policy_model = LLM(model=args.model_path, device=f'cuda:{local_rank}', gpu_memory_utilization=0.4, max_model_len=10000)

    pigai_model = PigaiModel.from_pretrained('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/experiments/pigai/v1/train_v2/checkpoint-467/').to(local_rank)
    pigai_model.eval()
    pigai_processor = AutoProcessor.from_pretrained("/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-1.5B-Instruct")

    # critique_model = CritiqueModel.from_pretrained(args.critique_model_path).to(policy_model.device)
    # critique_model.eval()
    critique_model = CritiqueModel.from_pretrained(args.critique_model_path).to(local_rank)
    critique_model.eval()
    critique_processor = AutoProcessor.from_pretrained("/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-7B-Instruct")

    basename = os.path.basename(args.dataset_path).split('.')[0]
    process_jsonl(rank, world_size, test_dataset, args.output_dir, policy_model, processor,critique_model, critique_processor, pigai_model, pigai_processor, basename, args)

    # dist.barrier()  # 这里进行同步，确保所有rank都完成了文件写入
    dist.destroy_process_group()

if __name__ == "__main__":
    # input_file = "/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/dataset/test_2_5vl.json"    # Path to input file
    # output_file = "/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/exp/qwen2.5vl/direct_critique/test_2_5vl.json"  # Path to output file
    
    # process_jsonl(input_file, output_file)
    # print(f"Processing complete! Results have been saved to {output_file}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/v1/experiments/self_play/debug')
    parser.add_argument("--model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-2B-Instruct') # policy model
    parser.add_argument("--pigai_model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-1.5B-Instruct') # pigai model
    parser.add_argument("--critique_model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-VL-3B-Instruct/') # PRM
    parser.add_argument("--dataset_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/datasets/mcts_dataset/test_val_all.json')
    parser.add_argument("--qwen2vl_infer_batch", type=int, default=1)
    parser.add_argument("--best_of_n", type=int, default=20) # BoN
    parser.add_argument("--top_p", type=float, default=0.9) # BoN
    parser.add_argument("--top_k", type=int, default=50) # BoN
    parser.add_argument("--temperature", type=float, default=0.7) # BoN
    parser.add_argument("--max_len", type=int, default=2048) # BoN
    parser.add_argument("--max_play_num", type=int, default=5)
    args = parser.parse_args()

    args.max_pixels = 800 * 600


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
