import os
import json
import hashlib
import time
import multiprocessing
from tqdm import tqdm

def get_unique_prefix(file_path):
    """基于文件路径生成唯一前缀"""
    abs_path = os.path.abspath(file_path)
    return hashlib.md5(abs_path.encode()).hexdigest()[:8]  # 取前8位哈希

def split_json_file(args):
    """拆分单个 JSON 文件"""
    file_path, output_dir, max_items_per_file = args
    # id_list = json.load(open('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/train_chqrtqa_m3cot_2_5/expand11_step30_Sims1_rollout16_t0.7_p0.9_k50/mcts_tree/id_list.json'))
    # id_list+=json.load(open('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/train_fillin_2_5/expand6_step30_Sims1_rollout10_t0.7_p0.9_k50/mcts_tree/id_list.json'))
    # id_list = [os.path.basename(idn).split('.pkl')[0] for idn in id_list]
    # id_list=set(id_list)
    if not os.path.isfile(file_path):
        return f"文件 {file_path} 不存在，跳过..."
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        return f"文件 {file_path} 不是包含列表的 JSON，跳过..."
    
    total_items = len(data)
    # print(f'{file_path}: {total_items}')
    num_parts = (total_items + max_items_per_file - 1) // max_items_per_file  # 计算拆分数量
    print(f'{file_path}: {total_items},max_items_per_file:{max_items_per_file} , num_parts:{num_parts}')

    prefix = get_unique_prefix(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_parts):
        chunk = data[i * max_items_per_file : (i + 1) * max_items_per_file]
        # chunk = [ch for ch in chunk if ch['id'].split('_cri')[0] not in id_list]
        output_path = os.path.join(output_dir, f"{prefix}_{base_name}_{timestamp}_part_{i+1}.json")
        if os.path.exists(output_path):
            output_path=output_path.replace('.json','_1.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=4)

    return f"Processed {file_path}, saved {num_parts} parts."

def parallel_split_json_files(file_paths, output_dir, max_items_per_file, num_workers=None):
    """使用多进程拆分多个 JSON 文件，带进度条"""
    os.makedirs(output_dir, exist_ok=True)

    task_args = [(fp, output_dir, max_items_per_file) for fp in file_paths]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(split_json_file, task_args), 
                            total=len(file_paths), desc="Processing JSON files"))

    for res in results:
        print(res)

# 示例用法
file_list = [#'/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/exp/v2_2/pigai_correct/5/train_fillin_current_neg_mathv360k.json',
                    # '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/exp/v2_2/pigai_correct/6/train_m3cot_choice_andrest_mathv360k.json',
                    # '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/exp/v2_2/pigai_correct/rest_all/rest_all.json',
                    # '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/rest/train_all_pos.json',
                    # '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/for_pack/qwen2.5vl/train_all_pos_rest.json',

                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/exp/v2_1/multi_revision/4/train_fillin_current_neg.json',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/qwen2vl/train_fillin_current_pos.json',
                    '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/for_pack/qwen2vl/train_fillin_current_pos_rest.json',

                    # '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/rest1/train_all_pos_rest1.json',
                    # '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/exp/v2_1/gen_critique/train_rest1/train_all_rest_neg.json',
                    # '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/rest1/train_all_pos.json'
                    ]  # JSON 文件列表
output_folder = "/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v2/dataset/split_qwen2vl"
max_items = 10000  # 每个 JSON 文件最多 500 条数据
num_workers = 8  # 4 个进程

parallel_split_json_files(file_list, output_folder, max_items, num_workers)
