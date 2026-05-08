import json
import os
import tqdm
import math
import numpy as np
import random
import pickle
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLProcessor
import concurrent.futures
import glob
# import ijson
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.extend(dir_path)
sys.path.append('/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/prm/v1.2/code/lmdb_utils')
from tools import LmdbWriter
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
from multiprocessing import Pool, cpu_count
def map_score(score):
    assert score>=0 and score<=1
    if score < 0.25:
        score = 1
    elif score < 0.5:
        score = 2
    elif score < 0.75:
        score = 3
    elif score < 0.9:
        score = 4
    else:
        score = 5
    return score
def split_list(lst, n=4):
    return [lst[i:i + n] for i in range(0, len(lst), n)]
def meta_func(file_path):
    # if 'm3cot' in cur_sample['id'].lower():
    #     return None, None
    diff_id_list = set(json.load(open('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/diff_id.json')))
    cur_datas = json.load(open(file_path))
    qwen2_flag=True
    if '/qwen2.5vl/' in file_path:
        # if 'mathv_' in file_path or 'm3cot' in file_path:
        if True:
            cur_data_1 = [da for da in cur_datas if da['id'].split('_cri')[0] in diff_id_list]
            cur_data_2 = [da for da in cur_datas if da['id'].split('_cri')[0] not in diff_id_list]
            cur_datas = random.sample(cur_data_2, int(len(cur_data_2)/4)) + cur_data_1
        qwen2_flag=False
    right_num=0
    wrong_num=0
    inputs_all=[]
    flag = False
  
    for cur_sample in cur_datas:
        raw_info = id_2_info[cur_sample['id'].split('_cri')[0].replace('qwen2vl_','')]
        # if cur_sample['id'] in train_id:
        image_grid_thw = raw_info['image_grid_thw']
        raw_input_tokens = raw_info['input_tokens']
        query = cur_sample['query'].replace('<|im_end|>', '')
        if 'e.g., A, B, C, D' in query and 'mathv360k' in cur_sample['id'].lower():
            prob = random.random()
            if prob<0.67:
                continue
        if 'cur_pigai_score' in cur_sample.keys():
            cur_score = cur_sample['cur_pigai_score']
        elif 'cur_score' in cur_sample.keys():
            cur_score = cur_sample['cur_score']
        else:
            if '_pos_' in cur_sample['id']:
                if 'selected_right' not in cur_sample.keys():
                    print(file_path)
                    break

                cur_score = cur_sample['selected_right'][-1]
            else:
                if 'selected_wrong' not in cur_sample.keys():
                    print(file_path)
                    break
                cur_score = cur_sample['selected_wrong'][-1]
            if not flag:
                print(file_path)
                flag=True
            # break
        if 'cri_pos' in cur_sample['id'] or '_revised' in cur_sample['id']:
            if map_score(cur_score)<3:
                continue
            original_answer = cur_sample['label']
        else:
            if map_score(cur_score)>=3:
                continue
            original_answer = (cur_sample['selected_wrong'][0]+cur_sample['selected_wrong'][1]).split('<|im_start|>assistant\n')[-1].replace('<|im_end|>', '')

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
        try:

            critique = cur_sample['critique'].replace('<|im_end|>', '')#+'<|im_end|>'
            critique = eval(critique)
            critique['score'] = map_score(cur_score)
            critique = str(critique)+'<|im_end|>'
        except:
            continue

        
        # input_text = user_prompt
        inputs = processor(
            text=user_prompt,
            images=None,
            videos=None,
            padding=True,
            return_tensors="np",
        )

        labels = processor(
            text=critique,
            images=None,
            videos=None,
            padding=True,
            return_tensors="np",
        )
        inputs['labels'] = np.array([[-100]*len(inputs['input_ids'][0]) + list(labels['input_ids'][0])])
        inputs['input_ids'] = np.array([list(inputs['input_ids'][0])+list(labels['input_ids'][0])])
        inputs['image_grid_thw'] = np.array(image_grid_thw, dtype=np.int32)
        inputs['pixel_name_string'] = os.path.basename(raw_info['pixel_values_path'])
        inputs['score'] = np.array(cur_score, dtype=np.float32)
        del inputs['attention_mask']
        if '_pos_' in cur_sample['id'] or '_revised_' in cur_sample['id']:
            right_num+=1
        else:
            wrong_num+=1
        
        inputs_all.append(inputs.data)
    # if ('aug_' in cur_sample['id'].lower() or 'human_' in cur_sample['id'].lower()) and '/qwen2.5vl/' not in file_path:
    #     inputs_all,right_num,wrong_num=inputs_all*3,right_num*3,wrong_num*3
    if 'human' in cur_sample['id'].lower() or 'aug_' in cur_sample['id'].lower():
        inputs_all,right_num,wrong_num=inputs_all*2,right_num*2,wrong_num*2
    return inputs_all, right_num, wrong_num, qwen2_flag


def chunk_lines(lines, num_chunks):
    chunk_size = len(lines) // num_chunks
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    if len(chunks) > num_chunks:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks


def process_lines_in_parallel(lines, func, multi_num=32):
    chunks = chunk_lines(lines, multi_num)
    with concurrent.futures.ThreadPoolExecutor(max_workers=multi_num) as executor:
        results, tokens = list(executor.map(func, chunks))
    processed_lines = [line for result in results for line in result if line is not None]
    processed_tokens = [line for result in tokens for line in result if line is not None]
    return processed_lines, processed_tokens

if __name__ == "__main__":
    # 原始数据
    raw_data_paths = ['/train34/cog8/permanent/bhwei2/pfhu6/shliu19/datasets/mcts_dataset/train_chartqa_m3cot_2_5.json', '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/datasets/mcts_dataset/train_all_2_5.json', '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/datasets/mcts_dataset/test_val_all_2_5.json', '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/datasets/mcts_dataset/train_choice_truefalse_2_5.json']


    

    base_paths = ['/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2.5vl/chartqa_neg',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2.5vl/chartqa_pos',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2.5vl/m3cot_neg',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2.5vl/m3cot_pos',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2.5vl/mathv_neg',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2.5vl/mathv_pos',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2vl/chartqa_neg',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2vl/chartqa_pos',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2vl/m3cot_neg',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2vl/m3cot_pos',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2vl/mathv_neg',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2vl/mathv_pos']
    file_paths=[]
    for base_path in base_paths:

        files = os.listdir(base_path)
        file_paths += [os.path.join(base_path, file) for file in files]

   # base_path = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v2/dataset/split_qwen2vl'
    #files = os.listdir(base_path)
    print('total_files num:', len(file_paths))


    output_dir = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/train_filter_choice/'


    output_path = os.path.join(output_dir, 'prompt.mdb')
    processor = AutoProcessor.from_pretrained('/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-7B-Instruct/')
    writer = LmdbWriter(output_path)
   
    raw_dataset = []
    for raw_data_path in raw_data_paths:
        with open(raw_data_path) as f:
            raw_dataset += json.load(f)
    id_2_info = {}
    for sample in raw_dataset:
        id_2_info[sample['id']] = sample
    raw_outputs=[]
    right_total=0
    wrong_total=0
    qwen2_num=0
    qwen2_5_num=0
    # file_paths=['/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v2/dataset/split_file/7b6fb433_train_m3cot_choice_andrest_mathv360k_20250320_031214_part_12.json']
    with Pool(80) as pool:
      for raw_output, right_num, wrong_num, qwen2_flag in tqdm.tqdm(pool.imap(meta_func, file_paths), total=len(file_paths)):
            if raw_output is not None:
                raw_outputs+=raw_output
                right_total+=right_num
                wrong_total+=wrong_num
                if qwen2_flag:
                    qwen2_num+=len(raw_output)
                else:
                    qwen2_5_num+=len(raw_output)

    print(f'负样例:{wrong_total}\n正样例:{right_total}')
    print(f'Qwen2-VL:{qwen2_num}\nQwen2.5-VL:{qwen2_5_num}')
    writer.write(raw_outputs)
    writer.flush()    
       
    writer.close()
    print('数据打包已完成')