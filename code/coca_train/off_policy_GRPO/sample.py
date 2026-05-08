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
import re
import json
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
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
sys.path.append('/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/')
from vision_processor import process_vision_info
from trl.trainer.utils import selective_log_softmax
from trl.extras.profiling import profiling_decorator
from vllm import LLM, SamplingParams
from torch.nn.utils.rnn import pad_sequence
from off_policy_GRPO.utils import PigaiAccuracyORM, FormatORM
import torch.nn.functional as F
my_env = os.environ.copy()  

my_env["PATH"] = "/home4/intern/ycpan4/miniconda3/envs/mcts_cp39/bin:/opt/lib/gcc-11.4.0/bin:/opt/lib/cuda-12.1/bin:" + my_env["PATH"] # 防止镜像没有module命令无法load gcc
os.environ['CUDA_HOME'] = '/opt/lib/cuda-12.1'
os.environ.update(my_env)
os.environ['TRITON_CACHE_DIR'] = '/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/Cache/.triton/autotune'
os.environ['CXX'] = 'g++'
os.environ["WANDB_DISABLED"] = 'true'
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


# def get_Qwen2_VL_input(processor, batch):
#     # print(batch)
#     messages = [ [{"role": "user", "content": [{"type": "image", "image": item['imgpath']}, {"type": "text", "text": "Describe this image."}]}] for item in batch ] # 处理图片
#     images, _ = process_vision_info(messages) # 得到pixel_values, image_grid_thw
#     text = [item['prompt'] for item in batch]
#     # Preparation for inference
#     inputs = processor(
#         text=text,
#         images=images,
#         videos=None,
#         padding=True,
#         return_tensors="pt",
#     )

#     return inputs



class GRPOSampler():
    def __init__(self, 
                actor,
                critic, 
                # engine,
                tokenizer, 
                processor, 
                test_dataset = None, 
                num_generation = 8, 
                top_p=0.9,
                top_k=50,
                temperature=0.7,
                max_len = 2048,
                qwen2vl_infer_batch = 1,
                args = None):
        self.actor = actor
        self.critic = critic
        # self.engine = engine
        self.processor = processor
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.num_generation = num_generation
        self.qwen2vl_infer_batch = qwen2vl_infer_batch

        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.max_len = max_len

        self.args = args
        self.device = self.critic.llm_engine.device_config.device
        # self.device = self.model.device
        self.accuracyORM = PigaiAccuracyORM(self.actor, self.processor, self.device)
        self.formatORM = FormatORM()
        self.rewards = [self.accuracyORM, self.formatORM]



    @profiling_decorator
    def _get_per_token_logps(self, model, inputs):
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        inputs = {
            k: v
            for k, v in inputs.items() if k not in
            ['logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps']
        }
        logits = model(**inputs).logits # [1, len, vocab_size]
        # exclude the last logit: it corresponds to the next token pred
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        logits = logits / self.temperature
        input_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens
    def Qwen2_VL_inference(self, query_list, imgpath_list, model, processor, cur_step_len):
        inputs = []
        for query, imgpath in zip(query_list, imgpath_list):
            messages = [{"role": "user", 
                        "content": [
                            {"type": "image", "image": imgpath}, 
                            {"type": "text", "text": query}]
                        }]
            text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )+'''Let's think step by step.'''
            images, videos = process_vision_info(messages)
            inputs += [{"prompt":text,"multi_modal_data": {"image": images[0]}}]
        samplingparams = SamplingParams(max_tokens=2048, temperature=0, skip_special_tokens=False)
        generated_ids_trimmed = []
        outputs = model.generate(inputs, samplingparams, use_tqdm=False)
        for out in outputs:
            generated_ids_trimmed.append(out.outputs[0].token_ids)#cumulative_logprob
            
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return output_text


    def Qwen2_VL_inference_batch(self, query_list, imgpath_list, model, processor, data_item):
        inputs = []
        for query, imgpath in zip(query_list, imgpath_list):
            messages = [
                {"role": "user", 
                        "content": [
                            {"type": "image", "image": imgpath}, 
                            {"type": "text", "text": query}]
                        }]
            text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )+'''Let's think step by step.'''
            images, videos = process_vision_info(messages)
            inputs += [{"prompt":text,"multi_modal_data": {"image": images[0]}}] * self.num_generation
        samplingparams = SamplingParams(max_tokens=2048, top_k=50, top_p=1.0, temperature=1.0, skip_special_tokens=False, repetition_penalty=1.05)
        generated_ids_trimmed = []
        outputs = model.generate(inputs, samplingparams, use_tqdm=False)
        for out in outputs:
            generated_ids_trimmed.append(out.outputs[0].token_ids)#cumulative_logprob
            
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

        answer_list = [comple.replace('<|im_end|>', '') for comple in output_text]
        question = data_item['question']
        target = data_item['target']
        reward, reward_list = self.accuracyORM.pigai_batch(question, answer_list, target, self.accuracyORM.pigai_model, self.accuracyORM.pigai_processor)

        if reward == 0 or reward == 1:
            return [answer_list[0], answer_list[1]], [reward_list[0], reward_list[1]], [], reward_list
        
        inputs = processor(
            text=text,
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        completions_prompt = [text + comple for comple in output_text]
        processor.tokenizer.padding_side = 'right'
        completion_inputs = processor(
            text=completions_prompt,
            images=images*len(completions_prompt),
            videos=None, 
            padding=True,
            return_tensors="pt",
        )
        labels = completion_inputs['input_ids'].clone()
        prompt_len = inputs['input_ids'].shape[1]
        labels[:, :prompt_len] = -100 # 将prompt部分值赋值为0
        # <|endoftext|>: 151643
        labels[labels == 151643] = -100 # 将所有<|endoftext|>处的值赋值为0
        logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
        # logits_to_keep = max([len(logp) for logp in old_per_token_logps])
        inputs = completion_inputs
        inputs['logits_to_keep'] = logits_to_keep # completion的最大长度
        inputs['completion_mask'] = labels[:, -logits_to_keep:] != -100

        ref_per_token_logps = torch.zeros((1,1)) # 如果后续需要添加KL散度的时候需要再算base model的KL散度
        old_per_token_logps = torch.zeros((1,1))
        rewards_per_func = torch.zeros((len(completions_prompt), 2))
        rewards_per_func[:, 0] = torch.tensor(reward_list, dtype=torch.float32)
        rewards_per_func[:, 1] = torch.tensor([1.0 if 'Final answer:' in complete else 0.0 for complete in output_text], dtype=torch.float32)

        # rewards = rewards_per_func.sum(dim=1)
        rewards = rewards_per_func[:,0] + rewards_per_func[:,1]*0.0
        
        # 计算advantages
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, len(completions_prompt)).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, len(completions_prompt)).std(dim=1)
        # 正则化, 来计算GRPO中的Advantages
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(len(completions_prompt), dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(len(completions_prompt), dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        inputs = inputs.to('cpu')
        advantages = advantages.to('cpu')
        outputs = [{
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'image_grid_thw': inputs['image_grid_thw'],
            'logits_to_keep': inputs['logits_to_keep'],
            'completion_mask': inputs['completion_mask'],
            'old_per_token_logps': old_per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'advantages': advantages,
            "rewards_per_func": rewards_per_func
        }]

        index_0 = reward_list.index(0)
        index_1 = reward_list.index(1)

        answer_list_sample = [answer_list[index_0], answer_list[index_1]]
        reward_list_sample = [0, 1]

        return answer_list_sample, reward_list_sample, outputs, reward_list
        # return answer_list[0], reward_list[0], outputs, reward_list

    def generate_critique(self, data_item, cur_step_len, expand_num):
        model = self.critic
        processor = self.processor
        query = data_item['query']

        original_answer = data_item['original_answer']
        # 一次推理num_generation次
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
        {"role": "user", "content": [{"type": "image", "image": data_item['imgpath']}, {"type": "text", "text": user_prompt}]}
    ]
        images, _ = process_vision_info(messages)
        inputs = processor(
            text=user_prompt,
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        
        prompt = user_prompt
        completions_prompt = []
        completions = []
        mm_inputs = [{"prompt":user_prompt,"multi_modal_data": {"image": images[0]}}]*expand_num
        samplingparams = SamplingParams(max_tokens=2048, top_p=1.0, top_k=50, temperature=1.0, skip_special_tokens=False, repetition_penalty=1.05)
        generated_ids_trimmed = []
        outputs = model.generate(mm_inputs, samplingparams, use_tqdm=False)
        for out in outputs:
            generated_ids_trimmed.append(out.outputs[0].token_ids)#cumulative_logprob
        completions = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

        completions_prompt = [prompt + comple for comple in completions]

        multi_images = images*len(completions_prompt)
        # 得到completions的input_ids, attention_mask, pixel_values, image_grid_thw
        processor.tokenizer.padding_side = 'right' # 设置<|end_of_text|>添加到右侧(默认为左侧)
        completion_inputs = processor(
            text=completions_prompt,
            images=multi_images,
            videos=None, 
            padding=True,
            return_tensors="pt",
        )
        completion_inputs = completion_inputs#.to(self.device)
        # 计算labels, 为后面计算completion部分的logps做准备, labels相当于是completion部分的mask
        labels = completion_inputs['input_ids'].clone()
        prompt_len = inputs['input_ids'].shape[1]
        labels[:, :prompt_len] = -100 # 将prompt部分值赋值为0
        # <|endoftext|>: 151643
        labels[labels == 151643] = -100 # 将所有<|endoftext|>处的值赋值为0
        return completions, completion_inputs, labels


    def getStepEnded(self, cul_steps):
        """
        获取解码到当前步骤是否已经解码结束.

        Returns:
            flag: True/False
        """
        tmp = cul_steps.strip()
        if tmp.count("<|im_end|>")==3  and not tmp.endswith("<|im_end|>"):
            print(tmp.replace('<|image_pad|>',''))
            print(self.imgpath)
        if tmp.endswith("<|im_end|>"):
            return True
        else:
            return False

    def sample(self, data_item):
        '''采样, 计算reward, Advantages, 并将采样结果保存到文件中'''
        device = self.device
        prompt = data_item['prompt']
        query = extract_prompt_content(prompt)
        data_item['query'] = query
        question = query.split('Question:')[-1].strip()
        # query = data_item['query']
        data_item['question'] = question
        # original_answer = '''Let's think step by step.''' + self.Qwen2_VL_inference([query], [data_item['imgpath']], self.actor, self.processor, data_item)[0]
        original_answer_, ori_reward_, actor_outputs_answer, ori_reward_list = self.Qwen2_VL_inference_batch([query], [data_item['imgpath']], self.actor, self.processor, data_item)


        outputs = []
        actor_outputs_revisions = []
        critic_reward_list = []
        actor_outputs_revision_reward_lists = []
        ori_rewards = []

        for original_answer, ori_reward in zip(original_answer_, ori_reward_):
            original_answer = '''Let's think step by step.''' + original_answer
            data_item['original_answer'] = original_answer
            # 推理
            completions, inputs, labels = self.generate_critique(data_item, cur_step_len=self.max_len, expand_num=self.num_generation)

            logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
            # logits_to_keep = max([len(logp) for logp in old_per_token_logps])
            inputs['logits_to_keep'] = logits_to_keep # completion的最大长度
            inputs['completion_mask'] = labels[:, -logits_to_keep:] != -100

            ref_per_token_logps = torch.zeros((1,1)) 
            old_per_token_logps = torch.zeros((1,1))

            # 计算reward
            rewards_per_func = torch.zeros((len(completions), len(self.rewards)), device=device)
            for i, reward_func in enumerate(self.rewards): # 使用reward fun或reward 
                if i==0:
                    output_reward_func, correction_identity, actor_outputs_revision, actor_outputs_revision_reward_list = reward_func(completions, ori_reward, **data_item)
                else:
                    output_reward_func = reward_func(completions, **data_item)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            actor_outputs_revisions += actor_outputs_revision
            actor_outputs_revision_reward_lists.append(actor_outputs_revision_reward_list)
            accuracy_reward = rewards_per_func[:, 0] # 第一个reward func为Accuracy Reward
            rewards_per_func = rewards_per_func
            # Apply weights to each reward function's output and sum
            # rewards = rewards_per_func.sum(dim=1)
            rewards = rewards_per_func[:,0] + rewards_per_func[:,1]*0.0
            
            # 计算advantages
            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, len(completions)).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, len(completions)).std(dim=1)
            # 正则化, 来计算GRPO中的Advantages
            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(len(completions), dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(len(completions), dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            

            inputs = inputs.to('cpu')
            advantages = advantages.to('cpu')
            outputs_cur = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'pixel_values': inputs['pixel_values'],
                'image_grid_thw': inputs['image_grid_thw'],
                'logits_to_keep': inputs['logits_to_keep'],
                'completion_mask': inputs['completion_mask'],
                'old_per_token_logps': old_per_token_logps,
                'ref_per_token_logps': ref_per_token_logps,
                'advantages': advantages,
                "rewards_per_func": rewards_per_func.to('cpu')

            }
            assert inputs['input_ids'].shape[0] == len(completions) == inputs['image_grid_thw'].shape[0] == self.num_generation
            if not(torch.all(accuracy_reward == 0) or torch.all(accuracy_reward == 1)):
                outputs.append(outputs_cur)
            critic_reward_list.append(accuracy_reward.cpu().tolist())
            ori_rewards.append(ori_reward)

        return outputs, critic_reward_list, ori_rewards, correction_identity, actor_outputs_answer, ori_reward_list, actor_outputs_revisions, actor_outputs_revision_reward_lists



    def sample_main(self, rank, world_size, local_rank):
        """
            两阶段对query进行采样, 
            stage1: vllm推理, 计算reward, advantages
            stage2: 使用pi_theta_old计算old_per_token_logps
        """
        test_dataset = self.test_dataset
        output_dir = self.args.output_dir
        test_dataset = test_dataset[rank::world_size] # 划分数据集
        # debug
        test_dataset = test_dataset[:] # TODO: 只取5个样本来debug
        logger.info(f'start rank_{rank}: len_queries: {len(test_dataset)}') # 划分数据集
        
        # ============= stage 1: vllm采样 =============
        # 增量保存配置
        sample_result = []
        pkl_save_path = os.path.join(self.args.output_dir, 'pkl')
        acc_save_path = os.path.join(self.args.output_dir, 'acc')

        pkl_actor_save_path = pkl_save_path
        pkl_critic_save_path = pkl_save_path

        # pkl_actor_save_path = os.path.join(pkl_save_path, 'actor')
        # pkl_critic_save_path = os.path.join(pkl_save_path, 'critic')
        for i, data_item in enumerate(test_dataset):
            save_id = data_item['id']
            save_path = os.path.join(pkl_critic_save_path, f'{save_id}_critic.pkl' )
            if os.path.exists(save_path):
                continue
            if i % 3 == 0:
                logger.info(f"rank: {rank}, sample progress: {i}/{len(test_dataset)}, {i/len(test_dataset)}")

            try:
                critic_outputs, accuracy_reward, ori_reward, correction_identity, actor_outputs_answer, ori_reward_list, actor_outputs_revision, actor_outputs_revision_reward_list= self.sample(data_item) # 推理
            except Exception as e:
                print(f"Error in sample, skip. Erro Info: {e}")
                continue
            save_id = data_item['id']
            save_path = os.path.join(acc_save_path, f'{save_id}.pkl' )
            with open(save_path, 'wb') as f:
                pickle.dump([[accuracy_reward, ori_reward, correction_identity], ori_reward_list, actor_outputs_revision_reward_list], f)
            # 保存pixel_values, 要不然会出错
            pixel_values_path = os.path.join('/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/dataset/datasets/mcts_dataset/pixel_values_train', os.path.basename(data_item['pixel_values_path']))
            # pixel_values_path = torch.load(os.path.join('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/datasets/mcts_dataset/pixel_values_train', os.path.basename(data_item['pixel_values_path'])))
            if not os.path.exists(pixel_values_path):
                print(f"Error in sample, skip. Erro Info: {pixel_values_path} not exists")
                continue
            # pixel_values_path = None
            for outputs in critic_outputs:
                save_data = [data_item, outputs]
                save_id = data_item['id']
                save_path = os.path.join(pkl_critic_save_path, f'{save_id}_critic.pkl' )


                save_data[1]['pixel_values'] = None
                save_data[0]['pixel_values_path'] = pixel_values_path
                

                with open(save_path, 'wb') as f:
                    pickle.dump(save_data, f)
           

        logger.info(f'rank: {rank} inference ends.')

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
    test_dataset = json.load(open(args.dataset_path)) # 读取数据集
    # test_dataset = random.sample(test_dataset, int(len(test_dataset)/2))
    logger.info(f'len_test_datasets: {len(test_dataset)}')
    

    pkl_save_path = os.path.join(args.output_dir, 'pkl')
    pkl_actor_save_path = pkl_save_path
    pkl_critic_save_path = pkl_save_path
    acc_save_path = os.path.join(args.output_dir, 'acc')
    # pkl_critic_save_path = os.path.join(pkl_save_path, 'critic')
    # pkl_actor_save_path = os.path.join(pkl_save_path, 'actor')
    if not os.path.exists(pkl_critic_save_path):
        os.makedirs(pkl_critic_save_path, exist_ok=True)
    if not os.path.exists(pkl_actor_save_path):
        os.makedirs(pkl_actor_save_path, exist_ok=True)
    if not os.path.exists(acc_save_path):
        os.makedirs(acc_save_path, exist_ok=True)
    # 清空pkl文件夹
    if rank == 0:
        pkl_dir = Path(pkl_critic_save_path)
        for pkl_file in pkl_dir.glob('*.pkl'):
            pkl_file.unlink() # 删除已经存在的pkl文件
        pkl_dir = Path(pkl_actor_save_path)
        for pkl_file in pkl_dir.glob('*.pkl'):
            pkl_file.unlink() # 删除已经存在的pkl文件


    test_dataset = test_dataset


    # 加载policy model, judge model
    # pretrained_path = "/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-7B-Instruct" # hu_debug
    pretrained_path = "/dmx-csy-mix01/cog3/permanent/qkchang/pretrained_models/Qwen2.5-VL-7B-Instruct/"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
    processor = AutoProcessor.from_pretrained(pretrained_path, max_pixels=args.max_pixels)

    critic = LLM(
        model=args.critic_model_path,
        device=f'cuda:{local_rank}',
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192*2,
        limit_mm_per_prompt={"image": args.num_generation, "video": args.num_generation},
        enforce_eager=False
        # max_num_seqs=256,
        # max_num_batched_tokens=8192
    )

    actor = LLM(
        model=args.actor_model_path,
        device=f'cuda:{local_rank}',
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192*2,
        limit_mm_per_prompt={"image": args.num_generation, "video": args.num_generation},
        enforce_eager=False
        # max_num_seqs=256,
        # max_num_batched_tokens=8192
    )

    sampler = GRPOSampler(actor = actor,
                        critic = critic, 
                        # engine = engine,
                        processor = processor, 
                        tokenizer = tokenizer, 
                        test_dataset = test_dataset, 
                        num_generation = args.num_generation,
                        top_p = args.top_p,
                        top_k = args.top_k,
                        temperature = args.temperature,
                        max_len = args.max_len,
                        qwen2vl_infer_batch = args.qwen2vl_infer_batch,
                        args = args)
    sampler.sample_main(rank, world_size, local_rank)
    # dist.barrier()  # 这里进行同步，确保所有rank都完成了文件写入
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
    parser.add_argument("--exp_id", type=str, default="debug")
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







