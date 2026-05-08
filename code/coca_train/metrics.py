import os
import sys
import json
import argparse
import heapq
import random

def split_dataset(dataset_path = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/datasets/mcts_dataset/test_val_all.json'):
    '''将数据集按照id类别进行划'''
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    
    test_dataset = json.load(open(dataset_path)) # 读取数据集
    
    # MMMU, M3CoT_test, M3CoT_val需要分开choice
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
                    'M3COT_test',
                    'M3COT_val',
                    'M3COT_test_choice',
                    'M3COT_val_choice',
                    'MathV360k',
                    'MATH_500']
    
    dataset_split = {key: [] for key in dataset_list}
    for test_item in test_dataset:
        item_id = test_item['id']
        find = False
        for dataset_name in dataset_list:
            if dataset_name in item_id:
                find = True
                if 'choice' in item_id:
                    cur_dataset_name = dataset_name + '_choice'
                else:
                    cur_dataset_name = dataset_name
                dataset_split[cur_dataset_name].append(test_item)
                break
        if not find:
            print(f"{item_id} not find")

    return dataset_split

def greedy_acc(result_path):
    print(f"Test on datset {result_path}")
    dataset_split = split_dataset(dataset_path=result_path)
    dataset_best_of_n_best = dataset_split
    
    # 统计Acc
    pigai_threshhold = 0.5
    dataset_acc = {key: {'True': [], 'False': []} for key in dataset_best_of_n_best} # 统计各个数据集的Acc
    for key, value in dataset_best_of_n_best.items():
        if len(value) > 0:
            for item in value:
                if 'pigai_score' in list(item.keys()):
                    if item['pigai_score'] >= pigai_threshhold: # 作答正确
                        dataset_acc[key]['True'].append(item)
                    else:
                        dataset_acc[key]['False'].append(item)
                else: # 没有评分默认作答错误
                    print(f"no pigai_score in {key}")
                    dataset_acc[key]['False'].append(item)

    # 统计各个数据集的Acc
    print("\n\n==========各数据集测评结果============")
    for dataset_name, dataset_pigai in dataset_acc.items():
        true_list = dataset_pigai['True']
        false_list = dataset_pigai['False']
        total_num = len(true_list) + len(false_list)
        total_num = max(total_num, 1) # 防止除数为0
        cur_acc = len(true_list) / total_num
        print(f"{dataset_name}: Acc: {cur_acc:.4f},  {len(true_list)}/{total_num}")
    print("============================\n\n")


def best_of_n_acc(result_path, best_of_n=2, passn=False):
    print(f"Test on datset {result_path}")
    dataset_split = split_dataset(dataset_path=result_path)
    dataset_best_of_n_best = {}
    # 筛选分数最高的结果
    
    for key, value in dataset_split.items():
        best_of_n_list = []
        if len(value) > 0:
            # 对value按照id进行排序, 确保相同id放在一起
            value = sorted(value, key = lambda x: x['id'])
            assert len(value) % best_of_n == 0
            for i in range(int(len(value) / best_of_n)):
                cur_bon_item = value[i * best_of_n: (i+1) * best_of_n]
                id_set = [item['id'] for item in cur_bon_item]
                assert len(set(id_set)) == 1 # 确保所有的N个都放在一起
                if not passn:
                    # 选出PRM分数最高的那个
                    PRM_value_list = [item['PRM_score'] for item in cur_bon_item]
                else:
                    # 天花板, bon中只要有一个对就算对, 直接当pigai_score当作PRM的分数
                    PRM_value_list = [item['pigai_score'] for item in cur_bon_item]

                max_index = PRM_value_list.index(max(PRM_value_list)) # 最好的结果

                # 随机选一个结果
                max_index = PRM_value_list.index(random.choice(PRM_value_list))
                
                # top_3 = heapq.nlargest(3, PRM_value_list)
                # max_value = random.choice(top_3)
                # max_index = PRM_value_list.index(max_value)

                # 保存最好的结果
                best_of_n_list.append(cur_bon_item[max_index])
        dataset_best_of_n_best[key] = best_of_n_list

    # print(dataset_best_of_n_best)
    # 统计Acc
    pigai_threshhold = 0.5
    dataset_acc = {key: {'True': [], 'False': []} for key in dataset_best_of_n_best} # 统计各个数据集的Acc
    for key, value in dataset_best_of_n_best.items():
        if len(value) > 0:
            for item in value:
                if 'pigai_score' in list(item.keys()):
                    if item['pigai_score'] >= pigai_threshhold: # 作答正确
                        dataset_acc[key]['True'].append(item)
                    else:
                        dataset_acc[key]['False'].append(item)
                else: # 没有评分默认作答错误
                    print(f"no pigai_score in {key}")
                    dataset_acc[key]['False'].append(item)

    # 统计各个数据集的Acc
    print("\n\n==========各数据集测评结果============")
    for dataset_name, dataset_pigai in dataset_acc.items():
        true_list = dataset_pigai['True']
        false_list = dataset_pigai['False']
        total_num = len(true_list) + len(false_list)
        total_num = max(total_num, 1) # 防止除数为0
        cur_acc = len(true_list) / total_num
        print(f"{dataset_name}: Acc: {cur_acc:.4f},  {len(true_list)}/{total_num}")
    print("============================\n\n")
    
def check_format(dataset_path):
    test_dataset = json.load(open(dataset_path)) # 读取数据集
    
    boxed_true_num = 0
    answer_think_true_num = 0
    for item in test_dataset:
        predict = item['predict']
        if 'boxed{' in predict:
            boxed_num = predict.count('boxed{')
            if boxed_num == 2:
                boxed_true_num += 1
        if '<answer>' in predict and '</answer>' in predict and '<think>' in predict and '</think>' in predict:
            if predict.count('<answer>')==3 and predict.count('</answer>')==3 and predict.count('<think>')==3 and predict.count('</think>')==3:
                answer_think_true_num += 1
    print(f"total num: {len(test_dataset)}")
    print(f"boxed_true_num: {boxed_true_num}")
    print(f"answer_think_true_num: {answer_think_true_num}")


def metric_pigai_acc(result_path, false_output_path=''):
    test_dataset = json.load(open(result_path)) # 读取数据集
    pigai_true_num = 0
    pigai_thresh = 0.45

    false_samples = []

    for item in test_dataset:
        if item['pigai_score'] >= pigai_thresh:
            cur_pigai_score = 1
        else:
            cur_pigai_score = 0
        
        if 'score' in item.keys():
            score = item['score']
        elif 'math_verify_score' in item.keys():
            score = item['math_verify_score']
        
        if score == cur_pigai_score:
            pigai_true_num += 1
        else:
            false_samples.append(item)
        


    pigai_acc = pigai_true_num / len(test_dataset)
    print(f"批改模型一致率: { pigai_acc }")
    
    if false_output_path != '':
        with open(false_output_path, 'w', encoding='utf-8') as f:
            json.dump(false_samples, f, indent=4, ensure_ascii=False)
        print(f"不一致样本保存到: {false_output_path}")


    

if __name__ == '__main__':
    bon_result_path = '/train21/cog8/permanent/qkchang/R1_Zero/experiments/infer_result/policy_result/Qwen_Math_template/BoN_8_[MATH_500]_2025041617/BoN_all_result_total.json'
    bon_result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/infer_result/policy_result/R1_template/BoN_8_[MATH_500]_2025041617/BoN_all_result_total.json"
    best_of_n = 8
    # best_of_n_acc(bon_result_path, best_of_n=best_of_n, passn=True)
    
    result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/infer_result/policy_result/R1_template/BoN_8_[MATH_500]_2025041617/BoN_all_result_total.json"
    # greedy_acc(result_path)
    
    
    # result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/infer_result/policy_result/Qwen_Math_template/BoN_8_[MATH_500]_2025041617/BoN_all_result_total.json"
    # result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/infer_result/policy_result/R1_template/BoN_8_[MATH_500]_2025041617/BoN_all_result_total.json"
    # check_format(result_path)
    
    
    # epoch_1, 2批改阈值选择 0.45, epoch_3阈值选择0.5
    result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/qwen_7b_pigai_ep2/piagi_[test_v1]_2025041810/BoN_all_result_total.json" # epoch_2 一致率 97.13
    result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/qwen_7b_pigai_ep1/piagi_[test_v1]_2025041810/BoN_all_result_total.json" # epoch_1 一致率 97.17
    result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/qwen_7b_pigai_ep3/piagi_[test_v1]_2025041810/BoN_all_result_total.json" # epoch_3 一致率 97.28
    

    result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/testset_v2_qwensample_7b_pigai_ep3/piagi_[test_v2_10k]_2025041814/BoN_all_result_total.json"
    result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/testset_v2_qwensample_7b_pigai_ep2/piagi_[test_v2_10k]_2025041814/BoN_all_result_total.json"
    result_path = "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/testset_v2_qwensample_7b_pigai_ep1/piagi_[test_v2_10k]_2025041814/BoN_all_result_total.json"

    false_output_path = result_path.replace('.json', '_false_samples.json')
    metric_pigai_acc(result_path, false_output_path)
    



