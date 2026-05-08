import json
import os
from collections import defaultdict
from tqdm import tqdm
from typing import List


def analysis_source(dataset: List[dict]):
    '''统计数据集中不同来源的分布'''
    source_list = [
        'LIMR',
        'T1',
        'asdiv',
        'gsm_8k',
        'math_12k',
        'math_lvl3to5',
        'orz',
        'deepscaler',
        'OREAL',
        'NuminaMath'
    ]
    source_num = {key: 0 for key in source_list}
    
    for item in tqdm(dataset):
        cur_id = item['id']

        is_find = False
        for source in source_list:
            if source in cur_id:
                source_num[source] += 1
                is_find = True
                break

        if not is_find:
            print(f"{cur_id} not find")
        
    for key, value in source_num.items():
        print(f"{key}:  {value}")




def get_query_distribution(dataset_path: str):
    '''获取query的难易分布'''
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 对数据集按照dataset排序
    # 对sample进行处理, 将相同id的sample放在同一个子list中, 以便区分搜索的不同步骤
    grouped = defaultdict(list)
    for item in dataset:
        grouped[item['id']].append(item)
    grouped_total_samples = list(grouped.values())

    # 记录所有的难易分布
    query_distribute = {'0': [], '1':[], '<0.5':[], '>0.5':[]}

    # # 记录各个数据集的难易分布
    # DATASET_QUERY_DISTRIBUTE = {}

    for group in tqdm(grouped_total_samples):
        cur_true_num = 0
        for item in group:
            if 'pigai_score' in item.keys():
                if item['pigai_score'] >= 0.5:
                    cur_true_num += 1
            else:
                continue
        cur_false_num = len(group) - cur_true_num
        cur_win_ratio = cur_true_num / len(group)
        # cur_item_list = [item for item in group]
        cur_item_list = [group[0]]
        if cur_win_ratio == 0:
            query_distribute['0'] += cur_item_list
        elif 0 < cur_win_ratio and cur_win_ratio <=0.5:
            query_distribute['<0.5'] += cur_item_list
        elif 0.5 < cur_win_ratio and cur_win_ratio < 1:
            query_distribute['>0.5'] += cur_item_list
        else:
            query_distribute['1'] += cur_item_list
        
    print(f"win_ratio == 0:  {len(query_distribute['0'])}")
    print(f"win_ratio < 0.5: {len(query_distribute['<0.5'])}")
    print(f"win_ratio > 0.5: {len(query_distribute['>0.5'])}")
    print(f"win_ratio == 1:  {len(query_distribute['1'])}")

    # 统计不同来源的分布
    print("================== win_ratio == 0 ==================")
    analysis_source(query_distribute['0'])
    print("================== win_ratio <0.5 ==================")
    analysis_source(query_distribute['<0.5'])
    print("================== win_ratio >0.5 ==================")
    analysis_source(query_distribute['>0.5'])
    print("================== win_ratio == 1 ==================")
    analysis_source(query_distribute['1'])
    


if __name__ == '__main__':
    
    dataset_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/RL_dataset/RL_all_dataset_bon_6.json"
    # dataset_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/RL_dataset_part_3/piagi_[tmp_total]_2025041817/BoN_all_result_total.json"
    get_query_distribution(dataset_path)


