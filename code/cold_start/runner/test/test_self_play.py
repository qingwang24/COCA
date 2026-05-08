import os
import sys
import json
import argparse
import re
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_file(file):
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
            # if record!='' and '{\n' == line.replace(' ',''):
            #     record=''
            # else:
            #     record+=line
            record+=line
            if not flag:
                records.append(json.loads(record))
                record = ''
    return records
def find_last_digit(s):
    match = re.findall(r'\d', s)  # 找到所有数字
    return match[-1] if match else None  # 返回最后一个数字
def split_dataset_and_score(dataset_path = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/datasets/mcts_dataset/test_val_all.json', output_dir = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/code/prm/v1/experiments/BoN/split', epoch=6, tokenizer=None):
    '''将数据集按照id类别进行划分并保存划分结果'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    test_dataset = json.load(open(dataset_path)) # 读取数据集
    id_set = set()
    # MMMU, M3CoT_test, M3CoT_val需要分开choice
    dataset_list = ['MathVision_mini',
                    'MathVision',
                    'MathVista_mini',
                    'mmmu_val',
                    'mmmu_val_choice',
                    'mmmu_pro_10',
                    'mmmu_pro_vision',
                    'test_aug', # charQA
                    'test_human', # charQA
                    'val_aug', # charQA
                    'val_human', # charQA
                    'train_human',
                    'train_aug',
                    'M3COT_test',
                    'M3COT_val',
                    'M3COT_train',
                    'M3COT_test_choice',
                    'M3COT_val_choice',
                    'MathV360k',
                    'ScienceQA',
                    'MathVerse',
                    'MMStar',
                    'LogicVista',
                    # 'HallusionBench',
                    'Dynamath',
                    'RealWorldQA',
                    'MMBench',
                    'MMTBench',
                    'SEED_Bench']
    
    num_conut = {key: 0 for key in dataset_list}
    dataset_split = {key: [] for key in dataset_list}
    dataset_acc = {key: {'True': [], 'False': []} for key in dataset_list} # 统计各个数据集的Acc
    dataset_acc_baseline = {key: {'True': [], 'False': []} for key in dataset_list}
    pigai_threshhold = 0.27 #HallBench
    pigai_threshhold = 0.92 #MathVista
    pigai_threshhold = 0.5 #chartqa
    pigai_threshhold_base=pigai_threshhold
    # pigai_threshhold_base=0.1
    # pigai_threshhold = 0.9
    # infer_threshhold_list = [0.8, 0.8, 0.8, 0.8, 0.65, 0.6]
    infer_threshhold_list = [0.8]*6
    mmmu_item_id_list = []
    false_item_id_list = []
    refine_count = 0
    token_len_all = []
    ori_token_len_all = []
    for test_item in test_dataset:
        # if 'choices' in test_item.keys():
        #     if test_item['choices'] is not None and test_item['choices']!=[]:
        #         continue
        item_id = test_item['id']
        if item_id in id_set:
            continue
        else:
            id_set.add(item_id)
        # if item_id.split('_cri')[0] not in id_list:
        #     continue
        find = False
        for dataset_name in dataset_list:
            if dataset_name in item_id:
                find = True
                if 'choice' in item_id:
                    cur_dataset_name = dataset_name + '_choice'
                else:
                    cur_dataset_name = dataset_name
                num_conut[cur_dataset_name] += 1
                dataset_split[cur_dataset_name].append(test_item)
                if 'pigai_score' in list(test_item.keys()):
                # if True:
                    # infer_sore_list = [test_item['pigai_score_infer']] + test_item['pigai_score_infer_list']
                    # if 'infer_score_all' in test_item.keys():
                    #     infer_sore_list = test_item['infer_score_all']
                    # else:
                    #     infer_sore_list = [0]*6
                    pigai_score_list = test_item['score_all']
                    if 'score_all_choice' in test_item.keys():
                        pigai_score_list_ = test_item['score_all_choice']
                    else:
                        pigai_score_list_=[]

                    answers_str_list = [test_item['original_answer']] + test_item['revision']
                    critique_str_list = test_item['critique']
                    critique_list = test_item['critique']
                    # epoch = 6
                    flag = False
                    select_idx = 0
                    cur_idx = -1
                    max_infer_sore = 0

                    select_idx=-1
                    cri_score_list=[]
                    output_str = ''
                    ori_str = test_item['original_answer']
                    for pigai_score, critique, infer_threshhold, answers_str, critique_str in zip(pigai_score_list[:epoch], critique_list[:epoch], infer_threshhold_list[:epoch], answers_str_list[:epoch], critique_str_list[:epoch]):
                        # try:
                        #     cri_score = eval(critique)['score']
                            
                        # except:
                        #     cri_score = None
                        cri_score = find_last_digit(critique)
                        cri_score_list.append(cri_score)
                        cur_idx += 1
                        
                        output_str += answers_str + critique_str
                        # if infer_score > infer_threshhold:
                        if (cri_score is not None and int(cri_score)>=5):
                        # if (cri_score is not None and int(cri_score)==3) and 'no corrections needed' not in critique.lower():
                        #     print(critique)
                        # if (cri_score is not None and int(cri_score)>=4) or infer_score > infer_threshhold:
                            flag=True
                            select_idx = cur_idx
                            break

                        select_idx+=1
                        # if infer_score>max_infer_sore:
                        #     max_infer_sore=infer_score
                        #     select_idx=cur_idx
                    
                    # if not flag:
                    #     select_idx=0
                    # else:
                    #     select_idx=random.int(1,4)
                    # score_l = [1 if s > 0.5 else 0 for s in pigai_score_list[1:]]
                    if tokenizer is not None:
                        ori_token_len_all.append(len(tokenizer(ori_str)['input_ids']))
                        token_len_all.append(len(tokenizer(output_str)['input_ids']))
                    pigai_score=pigai_score_list[select_idx]
                    # pigai_score=max(pigai_score_list)
                    pigai_score_base=pigai_score_list[0]

                    if 'test_human' in item_id:
                        pigai_score = pigai_score_list[0]
                    # if len(pigai_score_list_)==0:
                    #     pigai_score=pigai_score_list[0]
                    if 'MathVista' in item_id or 'MathVision' in item_id or 'MathVerse' in item_id or 'MMStar' in item_id or 'SEED_Bench' in item_id or 'RealWorldQA' in item_id or 'MMBench' in item_id or 'MMTBench' in item_id or 'mmmu' in item_id:
                        pigai_threshhold=0.92
                        pigai_threshhold_base=pigai_threshhold

                        # if 'problem_version' in test_item.keys():
                        #     if test_item['problem_version'] == "Vision Only":
                        #         continue
                        # pigai_score=max(pigai_score_list[select_idx],pigai_score_list[0])
                        if len(pigai_score_list_)>0:
                            # continue
                            pigai_score=max(pigai_score_list[select_idx],pigai_score_list_[select_idx])
                            # pigai_score=max(pigai_score_list[select_idx],pigai_score_list[select_idx_])

                            pigai_score_base=max(pigai_score_list[0],pigai_score_list_[0])
                        # else:
                        #     pigai_score = pigai_score_list[0]
                    else:
                        pigai_threshhold = 0.5
                        pigai_threshhold_base=pigai_threshhold

                    
                    # if select_idx == 2 and pigai_score>0.8:
                    #     print(test_item['query'])
                    #     print(critique_list[0])
                    #     print(critique_list[1])
                    #     1
                    # from PIL import Image
                    # if 'MathVista' in item_id and select_idx==epoch-2:
                    #     refine_count+=1
                    #     if pigai_score<0.5:
                    #         Image.open(test_item['imgpath']).save('/train21/cog8/permanent/qkchang/shliu19/critique/code/v4/analyze/1.png')
                    #         print(test_item['query'])
                    #         print(test_item['target'])
                    #         print(test_item['original_answer'])
                    #         print(1)
                    # if 'MathVista' in item_id and select_idx==epoch-1:
                    #     refine_count+=1
                    #     if pigai_score<0.5 and len(test_item['original_answer'])>1000:
                    #         Image.open(test_item['imgpath']).save('/train21/cog8/permanent/qkchang/shliu19/critique/code/v4/analyze/1.png')
                    #         print(test_item['query'])
                    #         print(test_item['target'])
                    #         print(test_item['original_answer'])
                    #         print(1)

                    # if infer_sore_list[0] > 0.6:
                    #     pigai_score=pigai_score_list[0]
                    # else:
                    #     if infer_sore_list[1] > 0:
                    #         # pigai_score=random.choice(pigai_score_list[1:])
                    #         pigai_score=max(random.sample(pigai_score_list, 1))
                    #         pigai_score=1 if sum(score_l)>= 3 else 0
                            # pigai_score=max(pigai_score_list[1:])
                    
                    # if pigai_score_list[0] > 0.5:
                    #     continue
                    # score_l = [1 if s > 0.5 else 0 for s in pigai_score_list[1:] ]
                    # if sum(score_l)>=5:

                    #     else:
                    #         pigai_score=pigai_score_list[0]
                        # pigai_score=max(pigai_score_list[1:])
                    
                    # pigai_score=max(pigai_score_list[:])



                    #召回
                    # pigai_score = pigai_score_list[0]
                    # if not pigai_score < 0.5:
                    #     continue
                    # # if cri_score is None or int(cri_score) < 3:
                    # infer_score = infer_sore_list[0]
                    # # if infer_score<=infer_threshhold:
                    # # if (cri_score_list[0] is None or int(cri_score_list[0]) < 4):
                    # if infer_score<=infer_threshhold and (cri_score_list[0] is None or int(cri_score_list[0]) < 3):

                    #精确
                    # if cri_score_list[0] is not None and int(cri_score_list[0])>=3:
                    #     continue
                    # infer_score = infer_sore_list[0]
                    # # if infer_score > 0.8:
                    # #     continue
                    # pigai_score = pigai_score_list[0]
                    # if pigai_score < 0.5:

                    #纠正
                    # pigai_score = pigai_score_list[0]
                    # if pigai_score > 0.5:
                    #     continue
                    # infer_score = infer_sore_list[0]
                    # if infer_score > infer_threshhold:
                    #     continue
                    # # if cri_score_list[0] is not None and int(cri_score_list[0])>=3:
                    # #     continue
                    # if pigai_score_list[1]>0.5:
                    # pigai_score=max(pigai_score_list[:])


                    # if test_item['target'].lower().replace(' ','') == test_item['original_answer'].lower().replace(' ',''):
                    if pigai_score > pigai_threshhold: # 作答正确
                        dataset_acc[cur_dataset_name]['True'].append(test_item)
                    else:
                        dataset_acc[cur_dataset_name]['False'].append(test_item)
                        false_item_id_list.append(item_id)

                    if pigai_score_base > pigai_threshhold_base:
                        dataset_acc_baseline[cur_dataset_name]['True'].append(test_item)
                    else:
                        dataset_acc_baseline[cur_dataset_name]['False'].append(test_item)
                # else: # 没有评分默认作答错误
                #     dataset_acc[cur_dataset_name]['False'].append(test_item)
                #     dataset_acc_baseline[cur_dataset_name]['False'].append(test_item)
                #     false_item_id_list.append(item_id)

                break
        # if not find:
        #     print(f"{item_id} not find")
    
    # print(f"数据集规模: {num_conut}")

    # for dataset_name, dataset in dataset_split.items():
    #     output_path = os.path.join(output_dir, dataset_name + '.json')
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         json.dump(dataset, f, indent=4)
    
    # 统计各个数据集的Acc
    print("\n\n==========各数据集测评结果============")
    true_list_all = 0
    total_num_all = 0
    true_list_all_baseline = 0
    total_num_all_baseline = 0
    chartqa_true_list=[]
    chartqa_total_num=0
    chartqa_true_list_baseline=[]
    chartqa_total_num_baseline=0
    for [dataset_name, dataset_pigai], [dataset_name_baseline, dataset_pigai_baseline] in zip(dataset_acc.items(), dataset_acc_baseline.items()):
        true_list = dataset_pigai['True']
        false_list = dataset_pigai['False']
        total_num = len(true_list) + len(false_list)
        total_num = max(total_num, 1) # 防止除数为0
        cur_acc = len(true_list) / total_num
        true_list_all += len(true_list)
        total_num_all += total_num if total_num>1 else 0

        true_list_baseline = dataset_pigai_baseline['True']
        false_list_baseline = dataset_pigai_baseline['False']
        total_num_baseline = len(true_list_baseline) + len(false_list_baseline)
        total_num_baseline = max(total_num_baseline, 1) # 防止除数为0
        cur_acc_baseline = len(true_list_baseline) / total_num_baseline
        true_list_all_baseline += len(true_list_baseline)
        total_num_all_baseline += total_num_baseline if total_num_baseline>1 else 0
        if len(true_list)<=1:
            continue
        print(f"{dataset_name}: Acc: {cur_acc:.4f},  {len(true_list)}/{total_num}")
        print(f"{dataset_name}: Acc: {cur_acc_baseline:.4f},  {len(true_list_baseline)}/{total_num_baseline}  Baseline")
        if dataset_name == 'test_aug':
            chartqa_true_list+=true_list
            chartqa_total_num+=total_num
            chartqa_true_list_baseline+=true_list_baseline
            chartqa_total_num_baseline+=total_num_baseline
        if dataset_name == 'test_human':
            chartqa_true_list+=true_list
            chartqa_total_num+=total_num
            chartqa_true_list_baseline+=true_list_baseline
            chartqa_total_num_baseline+=total_num_baseline
            print(f"test_chartqa: Acc: {len(chartqa_true_list) / chartqa_total_num:.4f},  {len(chartqa_true_list)}/{chartqa_total_num}")
            print(f"test_chartqa: Acc: {len(chartqa_true_list_baseline) / chartqa_total_num_baseline:.4f},  {len(chartqa_true_list_baseline)}/{chartqa_total_num_baseline}  Baseline")
    total_acc = true_list_all/total_num_all
    total_acc_baseline = true_list_all_baseline/total_num_all_baseline
    print(f"Total: Acc: {total_acc:.4f},  {true_list_all}/{total_num_all}")
    print(f"Total: Acc: {total_acc_baseline:.4f},  {true_list_all_baseline}/{total_num_all_baseline}  Baseline")
    if tokenizer is not None:
        print(f"average_token_len: {np.mean(token_len_all)}")
        print(f"ori_average_token_len: {np.mean(ori_token_len_all)}")
    print("============================\n\n")
    # print(refine_count)

    # with open(os.path.join(output_dir,'false_id_list.json'),'w',encoding='utf-8') as fi:
    #     json.dump(false_item_id_list, fi ,indent=4, ensure_ascii=False)




def main(args):
    combined_data = []
    all_json_files = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if (f.endswith('.json') and 'rank' in f)]
    
    basename = os.path.basename(all_json_files[0]).split('_rank')[0]
    for json_path in all_json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f) 
            combined_data += data
    BoN_total_save_path = os.path.join(args.output_dir, f"{basename}.json")
    if not os.path.exists(BoN_total_save_path):
        with open(BoN_total_save_path, 'w') as f:
            json.dump(combined_data, f, indent=4)

    for json_path in all_json_files:
        os.system(f"rm {json_path}") # 删除分rank

    
    # 划分数据集并保存
    split_dataset_and_score(dataset_path = BoN_total_save_path, output_dir = os.path.join(args.output_dir, 'split'))
    
    


if __name__ == '__main__':
    

    output_dir = '/train21/cog8/permanent/qkchang/shliu19/critique/code/v4/experiments/self_play/qwen2vl/test_final/'
    
    files = os.listdir(output_dir)
    files = [f for f in files if 'rank' in f]
    records_all = []
    if len(files) > 0:
        with Pool(8) as pool:
            for raw_output in tqdm(pool.imap(process_file, files), total=len(files)):
                records_all += raw_output
        print(len(records_all))
        # if len(records_all) == 1000:
        #     for file in files:
        #         file_path = os.path.join(output_dir, file)
        #         os.remove(file_path)

        dataset_path = os.path.join(output_dir, files[0]).split('_rank')[0]+'.json'
        with open(dataset_path, 'w', encoding='utf-8') as fi:
            json.dump(records_all, fi, ensure_ascii=False)
    else:
        files = os.listdir(output_dir)
        files = [f for f in files if '.json' in f]
        dataset_path = os.path.join(output_dir, files[0])

    output_dir = os.path.join(output_dir, 'split')

    tokenizer = None
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-7B-Instruct')
    epoch_list = [6,5,4,3,2]
    for epoch in epoch_list:
        print(f'epoch:{epoch-1}')
        split_dataset_and_score(dataset_path=dataset_path, output_dir=output_dir, epoch=epoch, tokenizer=tokenizer)
    # split_dataset_and_score(dataset_path='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/code/prm/v1/experiments/BoN/BoN_1_202502191958/BoN_total.json', output_dir='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/code/prm/v1/experiments/BoN/BoN_1_202502191958/split')





