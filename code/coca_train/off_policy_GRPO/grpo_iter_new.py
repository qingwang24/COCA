import os
import argparse
from datetime import datetime
import subprocess
import logging
logger = logging.getLogger(__name__)
import sys
from pathlib import Path
import json
import pandas as pd
import time
import json
import os
import pandas as pd
import pickle
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
def try_load(pklname):
    p = pklname
    try:
        with open(p, 'rb') as f:
            da = pickle.load(f)
        data_item = da[0]
        outputs = da[1]
        if outputs['old_per_token_logps'].shape[0] == 1:
            return None, None
        data_item['id'] = pklname.replace('.pkl', '')
        is_actor = 'actor' in pklname
        return data_item, is_actor
    except Exception:
        return None, None
occupy_shell = 'bash /dmx-csy-mix01/cog3/permanent/qkchang/R1_Zero/code/occupy/swift/z_script_occupy/grpo_sample_zk_qkchang/occupy.sh'


def split_dataset(dataset_path = '/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/datasets/mcts_dataset/test_val_all.json'):
    '''将数据集按照id类别进行划'''
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    
    test_dataset = json.load(open(dataset_path)) # 读取数据集
    
    # MMMU, M3CoT_test, M3CoT_val需要分开choice
    # dataset_list = ['MathVision_mini',
    #                 'MathVision',
    #                 'MathVista_mini',
    #                 'mmmu_val',
    #                 'mmmu_val_choice',
    #                 'test_aug', # charQA
    #                 'test_human', # charQA
    #                 'val_aug', # charQA
    #                 'val_human', # charQA
    #                 'train_human',
    #                 'M3COT_test',
    #                 'M3COT_val',
    #                 'M3COT_test_choice',
    #                 'M3COT_val_choice',
    #                 'MathV360k',
    dataset_list = ['MATH_500',
                    'AIME_2024',
                    'AIME_2025',
                    'LiveMathBench']
    
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



def greedy_acc(result_path, metric_output_path=None, exp_id=''):
    print(f"Test on datset {result_path}")
    dataset_split = split_dataset(dataset_path=result_path)
    dataset_best_of_n_best = dataset_split
    
    # 统计Acc
    pigai_threshhold = 0.5
    dataset_acc = {key: {'True': [], 'False': []} for key in dataset_best_of_n_best} # 统计各个数据集的Acc
    dataset_format_acc = {key: {'True': [], 'False': []} for key in dataset_best_of_n_best} # 统计各个数据集的Acc
    dataset_length = {key: {'True': [], 'False': []} for key in dataset_best_of_n_best} # 统计各个数据集的Acc
    for key, value in dataset_best_of_n_best.items():
        if len(value) > 0:
            for item in value:
                if 'pigai_score' in list(item.keys()):
                    if item['pigai_score'] >= pigai_threshhold: # 作答正确
                        dataset_acc[key]['True'].append(item)
                        if 'predict_length' in list(item.keys()):
                            dataset_length[key]['True'].append(item['predict_length'])
                    else:
                        dataset_acc[key]['False'].append(item)
                        if 'predict_length' in list(item.keys()):
                            dataset_length[key]['False'].append(item['predict_length'])
                else: # 没有评分默认作答错误
                    print(f"no pigai_score in {key}")
                    dataset_acc[key]['False'].append(item)
                    if 'predict_length' in list(item.keys()):
                        dataset_length[key]['False'].append(item['predict_length'])
                
                if 'format_score' in list(item.keys()):
                    if item['format_score'] == 1.0: # 格式正确
                        dataset_format_acc[key]['True'].append(item)
                    else: # 没有format_score默认格式错误
                        dataset_format_acc[key]['False'].append(item)


                
    metric = {}
    for key in dataset_acc.keys():
        metric[key] = {}
    # 统计各个数据集的Acc
    print("\n\n==========各数据集测评结果============")
    for dataset_name, dataset_pigai in dataset_acc.items():
        true_list = dataset_pigai['True']
        false_list = dataset_pigai['False']
        total_num = len(true_list) + len(false_list)
        total_num = max(total_num, 1) # 防止除数为0
        cur_acc = len(true_list) / total_num
        
        cur_format_true_list = dataset_format_acc[dataset_name]['True']
        cur_format_acc = len( cur_format_true_list) / total_num
        
        print(f"{dataset_name}: Acc: {cur_acc:.4f},  {len(true_list)}/{total_num}")
        print(f"{dataset_name}: Format_Acc: {cur_format_acc:.4f},  {len(cur_format_true_list)}/{total_num}")

        metric[dataset_name]['acc'] = cur_acc
        metric[dataset_name]['format_acc'] = cur_format_acc
        metric[dataset_name]['true_length'] = len(true_list)
        metric[dataset_name]['total_length'] = total_num
    print("============================\n\n")

    # 统计模型回复长度
    print("\n\n==========各数据集测评结果============")
    for dataset_name, dataset_length in dataset_length.items():
        true_list = dataset_length['True']
        false_list = dataset_length['False']
        total_list = true_list + false_list
        true_length = sum(true_list) / max(len(true_list), 1)
        false_length = sum(false_list) / max(len(false_list), 1)
        total_length = sum(total_list) / max(len(total_list), 1)
        print(f"{dataset_name}: True length: {true_length}, False length: {false_length}, Total length: {total_length}")
        metric[dataset_name]['true_predict_length'] = true_length
        metric[dataset_name]['false_predict_length'] = false_length
        metric[dataset_name]['total_predict_length'] = total_length
    
    print("============================\n\n")


    metric = {exp_id: metric}
    if metric_output_path:
        with open(metric_output_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(metric, ensure_ascii=False)  # 支持中文
            f.write(json_line + "\n")
    print(f"save output to {metric_output_path}")





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


def count_files_in_folder(folder_path):
    """
    计算指定文件夹下一级的文件数量（不包含子文件夹中的文件）。
    如果不存在当前文件夹, 认为0
    
    Args:
        folder_path (str): 要计算文件数量的文件夹路径。

    Returns:
        int: 文件夹中的文件数量。
    """
    if not os.path.exists(folder_path):
        return 0

    file_count = 0
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path) or os.path.isdir(full_path):
            file_count += 1
    return file_count


def monitor_folder(folder_path, duration_seconds=300, check_interval_seconds=10):
    """
    监控指定文件夹，当文件数量连续一段时间没有变化时发出提示。

    Args:
        folder_path (str): 要监控的文件夹路径。
        duration_seconds (int): 文件数量连续不变需要达到的秒数，达到此值触发提示（默认为 300 秒，即 5 分钟）。
        check_interval_seconds (int): 每次检查文件夹的时间间隔（默认为 10 秒）。
    """
    init_num = count_files_in_folder(folder_path)
    new_num = init_num
    while new_num <= init_num: # 直到开始出现新增文件时, 开始计时
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 等待新文件, 当前数量: {new_num} ")
        new_num = count_files_in_folder(folder_path)
        time.sleep(check_interval_seconds) # 等待一段时间再检查

    # 开始出现文件
    # 首次获取文件数量作为基准
    initial_file_count = count_files_in_folder(folder_path)
    last_file_count = initial_file_count
    no_change_start_time = time.time() # 记录文件数量最后一次变化的时间

    alert_triggered = False # 标志，用于避免重复触发提示

    print(f"[*] [{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在监控文件夹: {folder_path}")
    print(f"[*] [{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始文件数量: {initial_file_count}")
    print(f"[*] [{time.strftime('%Y-%m-%d %H:%M:%S')}] 每 {check_interval_seconds} 秒检查一次。")

    while not alert_triggered:
        time.sleep(check_interval_seconds) # 等待一段时间再检查
        current_time = time.time()

        try:
            current_file_count = count_files_in_folder(folder_path)

            if current_file_count > last_file_count:
                # 检测到文件数量增加 (新增文件)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测到新文件。文件数量从 {last_file_count} 增加到 {current_file_count}。")
                last_file_count = current_file_count
                no_change_start_time = current_time # 重置无变化计时
                alert_triggered = False # 文件发生变化，重置提示状态

            elif current_file_count == last_file_count:
                # 文件数量没有变化
                elapsed_time = current_time - no_change_start_time
                if elapsed_time >= duration_seconds and not alert_triggered:
                    # 文件数量连续不变的时间达到了设定的阈值，并且尚未触发提示
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 提示: 文件数量连续 {duration_seconds} 秒 ({duration_seconds/60:.0f} 分钟) 没有变化。当前文件数量为 {current_file_count}。")
                    alert_triggered = True # 标记提示已触发，避免重复输出


        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 发生未知错误: {e}")


def get_latest_checkpoint_id(model_output_dir: str) -> int:
    '''
        检测最新的checkpoint_id, 用来支持断点重提

        Args:
            model_output_dir: str, 模型文件输出路径
    '''

    checkpoint_id = 0 # -1表示从头开始训练
    exists_checkpoint_dirs = os.listdir(model_output_dir)
    # 提取数字
    # exists_checkpoint_dirs = ['iter_0', 'iter_4', 'iter_9', 'iter_10', 'iter_11']
    exists_iter_nums = []
    for item in exists_checkpoint_dirs:
        item_num = item.replace('iter_', '')
        try:
            item_num = int(item_num)
            exists_iter_nums.append(item_num)
        except Exception as e:
            pass
    
    exists_iter_nums = sorted(exists_iter_nums, reverse=True) # 逆序排序
    # 取出最大的iter_num
    for iter_num in exists_iter_nums:
        cur_iter_model_path = os.path.join(model_output_dir, f'iter_{str(iter_num)}')
        cur_iter_dirs = os.listdir(cur_iter_model_path)
        if len(cur_iter_dirs) > 2 and any(['checkpoint' in dir for dir in cur_iter_dirs]):
            # 至少存在checkpoint-xx 和 args.json
            checkpoint_id = iter_num # 找到
            break   

    if checkpoint_id == 0:
        return checkpoint_id
    else:
        return checkpoint_id+1



    
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='/train21/cog8/permanent/qkchang/R1_Zero/experiments/iter_debug/')
    parser.add_argument("--actor_model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-0.5B-Instruct/')
    parser.add_argument("--critic_model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-0.5B-Instruct/')
    parser.add_argument("--pigai_model_path", type=str, default='/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-0.5B-Instruct/') # pigai model
    parser.add_argument("--bash_path", type=str, default='/train34/cog8/permanent/bhwei2/pfhu6/qkchang/RL/code/swift/z_script_grpo/grpo_iter_qwen2.5_new/iter_new_v2') # 初始的数据集
    parser.add_argument("--test_dataset", type=str, default='/train21/cog8/permanent/qkchang/R1_Zero/dataset/Text_Only/benchmark/benchmark_debug.json')
    parser.add_argument("--num_generation", type=int, default=8) # G 
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--iter_num", type=int, default=3)
    parser.add_argument("--begin_id", type=int, default=0) # 开始迭代的id, 便于中间迭代时报错重新启动
    parser.add_argument("--deepspeed", type=str, default='zero1')
    parser.add_argument("--skip_first_sample", type=str, default='False') # 
    parser.add_argument("--check_interval_seconds", type=int, default=2)
    parser.add_argument("--duration_seconds", type=int, default=5)
    parser.add_argument("--task_name", type=str, default='debug')
    

    parser.add_argument("--NGPUS", type=int, default=4)
    parser.add_argument("--NNODES", type=int, default=1)
    parser.add_argument("--MASTER_PORT", type=int, default=10025)
    args = parser.parse_args()
    args.model_output_dir = os.path.join(args.output_dir, 'model_output')
    args.sample_output_dir = os.path.join(args.output_dir, 'sample_output')
    args.dataset_output_dir = os.path.join(args.output_dir, 'dataset_output')
    args.infer_output_dir = os.path.join(args.output_dir, 'infer_output')
    args.log_output_dir = os.path.join(args.output_dir, 'run_log')

    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.log_output_dir):
        os.makedirs(args.log_output_dir, exist_ok=True)

    def handle_exception(exc_type, exc_value, exc_tb):
        # 如果是系统退出异常，忽略
        if exc_type == SystemExit:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        # 记录异常到日志
        logging.error("未捕获的异常", exc_info=(exc_type, exc_value, exc_tb))
    # 设置 sys.excepthook 捕获未处理的异常
    sys.excepthook = handle_exception


    rank = 0
    node_id = rank // 8  # 假设每个节点有 8 张卡
    logging.basicConfig(
        format="Node[{}] %(asctime)s - %(levelname)s - %(message)s".format(rank),
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(f'{args.output_dir}/run_log/run-debug-node[{node_id}].log'), mode='a'),
            logging.StreamHandler(sys.stdout)
            ]
    )
    logger.setLevel(logging.INFO)
    

    # main(args)
    sample_output_dir = args.sample_output_dir
    model_output_dir = args.model_output_dir
    actor_model_output_dir = os.path.join(model_output_dir, 'actor')
    critic_model_output_dir = os.path.join(model_output_dir, 'critic')
    dataset_output_dir = args.dataset_output_dir
    infer_output_dir = args.infer_output_dir
    actor_model_path = args.actor_model_path # 初始的模型路径
    critic_model_path = args.critic_model_path
    pigai_model_path = args.pigai_model_path
    # source_dataset_path = args.source_dataset_path
    num_generation = args.num_generation
    # query_distribution = args.query_distribution
    test_dataset = args.test_dataset
    
    
    # 采样参数
    top_p=1.0
    top_k=50
    temperature=0.7
    iter_num=10000
    max_len=4096
    infer_max_len=8192
    
    # GRPO训练参数
    begin_id=args.begin_id
    deepspeed=args.deepspeed
    kl_beta=0
    lr=1e-6
    gradient_accumulation_steps=args.gradient_accumulation_steps
    iter_nums=args.iter_num
    gpu_memory_utilization=args.gpu_memory_utilization
    per_device_train_batch_size=args.per_device_train_batch_size
    # 分布式训练参数
    NGPUS = args.NGPUS
    NNODES = args.NNODES
    MASTER_PORT = args.MASTER_PORT
    gradient_accumulation_steps = int(2048/NNODES/NGPUS/per_device_train_batch_size)
    

    # ===============================================================
    # 断点重提: 检测最新的check_point id
    # ===============================================================
    # try:
    #     begin_id = get_latest_checkpoint_id(model_output_dir)
    # except Exception as e:
    #     begin_id = 0
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测到最新的checkpoint {begin_id}, 从该checkpoint恢复训练")
    
    begin_id = args.begin_id
    # debug
    # for iter_id in range(begin_id, iter_nums):
    test_dataset_ori = json.load(open(test_dataset))
    total_len = len(test_dataset_ori)
    slice_len = int(total_len / 4)
    import random
    for iter_id in range(begin_id, begin_id + iter_nums):
        test_dataset = args.test_dataset
        test_dataset_sample = test_dataset_ori[iter_id%4 * slice_len: (iter_id%4 + 1) * slice_len]
        test_dataset = test_dataset.replace('.json', f'_sample_{iter_id%4}.json')
        if not os.path.exists(test_dataset):
            with open(test_dataset, 'w') as fi:
                json.dump(test_dataset_sample, fi)
        dataset_path = os.path.join(dataset_output_dir, f'iter_{iter_id}.json')

        
        # ==================================================================
        #          Step: 1 数据采样
        # ==================================================================
        now = datetime.now()
        current_time_str = now.strftime("%Y%m%d%H%M%S")
        logger.info("==========================================")
        logger.info(f"============{current_time_str} 开始第 {iter_id} 次数据采样============")
        logger.info("==========================================\n\n\n")
        sample_output_dir = sample_output_dir
        infer_output_dir = infer_output_dir
                    # f"--test_dataset {test_dataset}" + " " + \
                    # f"--infer_output_dir {infer_output_dir}" + " " + \
                    # f"--infer_max_len {infer_max_len}" + " " + \
        if NNODES > 1:
            sample_command = f"torchrun --nproc_per_node {NGPUS} --nnodes={NNODES} --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port={MASTER_PORT} "
        else:
            sample_command = f"torchrun --nproc_per_node {NGPUS} --master_port={MASTER_PORT} "
        sample_args = \
            f"./off_policy_GRPO/sample.py" + " " + \
            f"--output_dir {sample_output_dir}" + " " + \
            f"--actor_model_path {actor_model_path}" + " " + \
            f"--critic_model_path {critic_model_path}" + " " + \
            f"--pigai_model_path {pigai_model_path}" + " " + \
            f"--dataset_path {test_dataset}" + " " + \
            f"--qwen2vl_infer_batch 1" + " " + \
            f"--num_generation {num_generation}" + " " + \
            f"--top_p {top_p}" + " " + \
            f"--top_k {top_k}" + " " + \
            f"--temperature {temperature}" + " " + \
            f"--max_len {max_len}" + " " + \
            f"--gpu_memory_utilization {gpu_memory_utilization}" + " " + \
            f"--exp_id sample_iter_{iter_id}"
        
        sample_command += sample_args

        logger.info(sample_command)
        # 生成stage1_sample.sh
        stage1_template_path = os.path.join(os.path.join(args.bash_path, 'template'), 'stage1_sample_template.sh')
        stage1_sample_path = os.path.join(args.bash_path, 'stage1_sample.sh')
        with open(stage1_template_path, 'r', encoding='utf-8') as f:
            stage1_template = f.readlines()
        stage1_template.append(sample_command)
        with open(stage1_sample_path, 'w', encoding='utf-8') as f:
            f.writelines(stage1_template)
        stage1_sample_submit_path = stage1_sample_path.replace('.sh', '_submit.sh')

        
        cur_iter_sample_path = os.path.join(sample_output_dir, f"sample_iter_{iter_id}")
        grpo_train_dataset_path = os.path.join(cur_iter_sample_path, "dataset")
        cur_iter_model_output_dir = os.path.join(model_output_dir, f"iter_{iter_id}")
        pkl_save_path = os.path.join(grpo_train_dataset_path, 'pkl')

        grpo_train_dataset_path_actor = os.path.join(grpo_train_dataset_path, "actor")
        grpo_train_dataset_path_critic = os.path.join(grpo_train_dataset_path, "critic")
        cur_iter_model_output_dir_actor = os.path.join(cur_iter_model_output_dir, f"actor")
        cur_iter_model_output_dir_critic = os.path.join(cur_iter_model_output_dir, f"critic")
        # pkl_actor_save_path = os.path.join(pkl_save_path, 'actor')
        # pkl_critic_save_path = os.path.join(pkl_save_path, 'critic')
        pkl_actor_save_path = pkl_save_path
        pkl_critic_save_path = pkl_save_path
        pkl_critic_save_path_new = os.path.join(grpo_train_dataset_path, 'pkl_new')
        acc_save_path = os.path.join(grpo_train_dataset_path, 'acc')
        if not os.path.exists(pkl_critic_save_path_new):
            os.makedirs(pkl_critic_save_path_new)
        if not os.path.exists(acc_save_path):
            os.makedirs(acc_save_path)
        if not os.path.exists(pkl_save_path):
            os.makedirs(pkl_save_path) 
        if not os.path.exists(grpo_train_dataset_path):
            os.makedirs(grpo_train_dataset_path) 
        if not os.path.exists(cur_iter_model_output_dir):
            os.makedirs(cur_iter_model_output_dir) 
        
        
        # ===============================================================
        # 断点重提: 判断是否已经生成了dataset.json, 如果生成, 就跳过采样阶段
        # ===============================================================
        final_dataset_save_path = os.path.join(cur_iter_sample_path, 'dataset_critic.json')
        if os.path.exists(final_dataset_save_path):
            # 检查是否有内容
            try:
                with open(final_dataset_save_path, 'r', encoding='utf-8') as f:
                    _tmp_dataset = json.load(f)
                if len(_tmp_dataset) > 0:
                    args.skip_first_sample = 'True'
                else:
                    args.skip_first_sample = 'False'
            except:
                args.skip_first_sample = 'False'
        else:
            args.skip_first_sample = 'False'
        


        # ==================================================================
        #          Submit: 1 数据采样提交
        # ==================================================================
        # kill占卡脚本
        if args.skip_first_sample != 'True' and iter_id!=-1:
            # pass
            # debug
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(5)
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(5)
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(5)
            
            # 提交stage1_sample.sh
            print(f'bash {stage1_sample_submit_path} {NGPUS} {NNODES}')
            os.system(f'bash {stage1_sample_submit_path} {NGPUS} {NNODES}')
            time.sleep(5)
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(30) # 等待提交成功

            os.system(occupy_shell)
            time.sleep(10)
            os.system(occupy_shell)


            # 判断stage1_sample是否结束, pkl文件夹出现文件之后连续5分钟不变化认为结束
            monitor_folder(pkl_critic_save_path, duration_seconds=args.duration_seconds, check_interval_seconds=args.check_interval_seconds)


        # ==================================================================
        #          Submit: 1 数据采样完成
        # ==================================================================


        # ==================================================================
        #          Step: 1 数据采样后处理
        # ==================================================================
        now = datetime.now()
        current_time_str = now.strftime("%Y%m%d%H%M%S")
        logger.info("==========================================")
        logger.info(f"============{current_time_str} 开始第 {iter_id} 次数据采样后处理============")
        logger.info("==========================================\n\n\n")
        sample_output_dir = sample_output_dir
        infer_output_dir = infer_output_dir
        if NNODES > 1:
            sample_command = f"torchrun --nproc_per_node {NGPUS} --nnodes={NNODES} --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port={MASTER_PORT} "
        else:
            sample_command = f"torchrun --nproc_per_node {NGPUS} --master_port={MASTER_PORT} "
        sample_args = \
            f"./off_policy_GRPO/sample_post_process.py" + " " + \
            f"--output_dir {sample_output_dir}" + " " + \
            f"--actor_model_path {actor_model_path}" + " " + \
            f"--critic_model_path {critic_model_path}" + " " + \
            f"--pigai_model_path {pigai_model_path}" + " " + \
            f"--dataset_path {test_dataset}" + " " + \
            f"--qwen2vl_infer_batch 1" + " " + \
            f"--num_generation {num_generation}" + " " + \
            f"--top_p {top_p}" + " " + \
            f"--top_k {top_k}" + " " + \
            f"--temperature {temperature}" + " " + \
            f"--max_len {max_len}" + " " + \
            f"--gpu_memory_utilization {gpu_memory_utilization}" + " " + \
            f"--exp_id sample_iter_{iter_id}"
        
        sample_command += sample_args

        logger.info(sample_command)
        # 生成stage1_sample.sh
        stage1_template_path = os.path.join(os.path.join(args.bash_path, 'template'), 'stage1_sample_template.sh')
        stage1_sample_path = os.path.join(args.bash_path, 'stage1_sample_post.sh')
        with open(stage1_template_path, 'r', encoding='utf-8') as f:
            stage1_template = f.readlines()
        stage1_template.append(sample_command)
        with open(stage1_sample_path, 'w', encoding='utf-8') as f:
            f.writelines(stage1_template)
        stage1_sample_submit_path = stage1_sample_path.replace('.sh', '_submit.sh') 
        
        # ==================================================================
        #          Submit: 1 数据采样后处理提交
        # ==================================================================
        # kill占卡脚本
        if args.skip_first_sample != 'True' and iter_id!=-1:
            # pass
            # debug
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(5)
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(5)
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(5)
            
            # 提交stage1_sample.sh
            print(f'bash {stage1_sample_submit_path} {NGPUS} {NNODES}')
            os.system(f'bash {stage1_sample_submit_path} {NGPUS} {NNODES}')
            time.sleep(5)
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(30) # 等待提交成功

            os.system(occupy_shell)
            time.sleep(10)
            os.system(occupy_shell)


            # 判断stage1_sample_post是否结束, pkl文件夹出现文件之后连续5分钟不变化认为结束
            monitor_folder(pkl_critic_save_path_new, duration_seconds=90, check_interval_seconds=args.check_interval_seconds)


        # ==================================================================
        #          Submit: 1 数据采样后处理完成
        # ==================================================================







        # Step 1.5数据集生成以及metric计算
        # 生成dataset.json
        logger.info("=======================\n Generate TrainDataset \n=======================")
        if iter_id!=-1 and args.skip_first_sample != 'True':
            pkl_file_list = os.listdir(pkl_critic_save_path_new)
            final_pkl_list = [os.path.join(pkl_critic_save_path_new, f) for f in pkl_file_list if f.endswith('.pkl')]
            final_pkl_list = set(final_pkl_list)
            final_dataset_save_path_critic = os.path.join(cur_iter_sample_path, 'dataset_critic.json')
            final_dataset_save_path_actor = os.path.join(cur_iter_sample_path, 'dataset_actor.json')
            final_dataset_critic = []
            final_dataset_actor = []

            # chunksize = max(1, len(final_pkl_list) // 20)
            # results = process_map(try_load,
            #                         final_pkl_list,
            #                         max_workers=20,    # 并行进程数
            #                         desc='filter dataset', chunksize=chunksize)
            # with Pool(20) as pool:
            #     # 使用tqdm显示进度条
            #     results = list(tqdm(
            #         pool.imap(try_load, final_pkl_list),
            #         total=len(final_pkl_list),
            #         desc="Processing files",
            #         unit="file"
            #     ))
            for pkl_path in tqdm(final_pkl_list): 
                data_item, is_actor = try_load(pkl_path)
            # for data_item, is_actor in results:
                if data_item is None:
                    continue
                if is_actor:
                    final_dataset_actor.append(data_item)
                else:
                    final_dataset_critic.append(data_item)

            with open(final_dataset_save_path_critic, 'w', encoding='utf-8') as f:
                json.dump(final_dataset_critic, f, ensure_ascii=False, indent=4)

            json_to_parquet(final_dataset_save_path_critic, grpo_train_dataset_path_critic)

            with open(final_dataset_save_path_actor, 'w', encoding='utf-8') as f:
                json.dump(final_dataset_actor, f, ensure_ascii=False, indent=4)

            json_to_parquet(final_dataset_save_path_actor, grpo_train_dataset_path_actor)
        
        




        # ==================================================================
        #          Step: 2 GRPO训练
        # ==================================================================
        now = datetime.now()
        current_time_str = now.strftime("%Y%m%d%H%M%S")
        logger.info("==========================================")
        logger.info(f"============{current_time_str} 开始第 {iter_id} 次GRPO训练============")
        logger.info("==========================================\n\n\n")
        if NNODES > 1:
            grpo_command = f"torchrun --nproc_per_node {NGPUS} --nnodes={NNODES} --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port={MASTER_PORT} "
        else:
            grpo_command = f"torchrun --nproc_per_node {NGPUS} --master_port={MASTER_PORT} "
        grpo_args = \
            f"./swift/cli/rlhf.py" + " " + \
            f"--rlhf_type grpo" + " " + \
            f"--model {critic_model_path}" + " " + \
            f"--external_plugins ./examples/train/grpo/plugin/plugin.py" + " " + \
            f"--reward_funcs format" + " " + \
            f"--use_vllm false" + " " + \
            f"--vllm_gpu_memory_utilization {gpu_memory_utilization}" + " " + \
            f"--temperature {temperature}" + " " + \
            f"--top_p {top_p}" + " " + \
            f"--top_k {top_k}" + " " + \
            f"--train_type full" + " " + \
            f"--torch_dtype bfloat16" + " " + \
            f"--dataset {grpo_train_dataset_path_critic}" + " " + \
            f"--split_dataset_ratio 0" + " " + \
            f"--max_completion_length {max_len}" + " " + \
            f"--num_train_epochs 1" + " " + \
            f"--per_device_train_batch_size {per_device_train_batch_size}" + " " + \
            f"--per_device_eval_batch_size 1" + " " + \
            f"--learning_rate {lr}" + " " + \
            f"--warmup_ratio 0" + " " + \
            f"--lr_scheduler_type constant" + " " + \
            f"--gradient_accumulation_steps {gradient_accumulation_steps}" + " " + \
            f"--eval_strategy no" + " " + \
            f"--save_strategy epoch" + " " + \
            f"--save_total_limit 10" + " " + \
            f"--logging_steps 3" + " " + \
            f"--max_length {max_len}" + " " + \
            f"--output_dir {cur_iter_model_output_dir_critic}" + " " + \
            f"--dataloader_num_workers 4" + " " + \
            f"--dataset_num_proc 4" + " " + \
            f"--num_generations {num_generation}" + " " + \
            f"--repetition_penalty 1.05" + " " + \
            f"--system ./prompt.txt" + " " + \
            f"--log_completions true" + " " + \
            f"--num_iterations 1" + " " + \
            f"--async_generate false" + " " + \
            f"--beta {kl_beta}" + " " + \
            f"--add_version False" + " " + \
            f"--model_type qwen2_5_vl" + " " + \
            f"--create_checkpoint_symlink True" + " " + \
            f"--freeze_vit True"
            # f"--max_grad_norm 1.0"

        grpo_command += grpo_args

        
        logger.info(grpo_command)
        # 生成stage1_sample.sh
        stage2_template_path = os.path.join(os.path.join(args.bash_path, 'template'), 'stage2_train_template.sh')
        stage2_train_path = os.path.join(args.bash_path, 'stage2_train.sh')
        with open(stage2_template_path, 'r', encoding='utf-8') as f:
            stage2_template = f.readlines()
        stage2_template.append(grpo_command)
        with open(stage2_train_path, 'w', encoding='utf-8') as f:
            f.writelines(stage2_template)
        stage2_train_submit_path = stage2_train_path.replace('.sh', '_submit.sh')


        # ==================================================================
        #          Submit: 2 训练任务提交
        # ==================================================================
        if iter_id!=-1:
            train_begin = False
            
            while not train_begin:
            
                # debug
                os.system('yes | ky exp delete -n train-h200-v000')
                time.sleep(5)
                os.system('yes | ky exp delete -n train-h200-v000')
                time.sleep(5)
                os.system('yes | ky exp delete -n train-h200-v000')
                time.sleep(5)
                
                os.system(f'bash {stage2_train_submit_path} {NGPUS} {NNODES} {args.task_name}')
                time.sleep(5)
                os.system('yes | ky exp delete -n train-h200-v000')
                time.sleep(30) # 等待提交成功


                os.system(occupy_shell)
                time.sleep(10)
                os.system(occupy_shell)
                
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GRPO 训练任务提交完成")
                # 监控是否生成了logging.jsonl
                logging_jsonl_path = os.path.join(cur_iter_model_output_dir_critic, 'logging.jsonl')
                print(f"[*] [{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始监控 {logging_jsonl_path}, 等待600s")
                time.sleep(600) # 等待20min, 60*20=1200
                if os.path.exists(logging_jsonl_path):
                    train_begin = True
                else:
                    train_begin = False # 没有启动训练, 再次提交
                    # # kill训练任务
                    train_task_name = f'train-grpo-shl-{args.task_name}'
                    os.system(f'yes | ky exp delete -n {train_task_name}')
                    time.sleep(5)
                    os.system(f'yes | ky exp delete -n {train_task_name}')
                    time.sleep(5)
                    os.system(f'yes | ky exp delete -n {train_task_name}')

                    print(f"[*] [{time.strftime('%Y-%m-%d %H:%M:%S')}] 没有启动GRPO训练, 再次重提")



            # 判断stage2_train是否结束, 监控model_output/iter_{i}/images
            monitor_folder(os.path.join(cur_iter_model_output_dir_critic, 'images'), duration_seconds=int(args.duration_seconds / 2), check_interval_seconds=args.check_interval_seconds)
        
        # ==================================================================
        #          Step: 2 GRPO训练完成
        # ==================================================================
        
        
        # 更新model_path
        critic_model_path = os.path.join(cur_iter_model_output_dir_critic, 'last')
        
        args.skip_first_sample = 'False'
        
        # ==================================================================
        #          Step: 3 GRPO训练
        # ==================================================================
        now = datetime.now()
        current_time_str = now.strftime("%Y%m%d%H%M%S")
        logger.info("==========================================")
        logger.info(f"============{current_time_str} 开始第 {iter_id} 次GRPO训练============")
        logger.info("==========================================\n\n\n")
        if NNODES > 1:
            grpo_command = f"torchrun --nproc_per_node {NGPUS} --nnodes={NNODES} --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port={MASTER_PORT} "
        else:
            grpo_command = f"torchrun --nproc_per_node {NGPUS} --master_port={MASTER_PORT} "
        grpo_args = \
            f"./swift/cli/rlhf.py" + " " + \
            f"--rlhf_type grpo" + " " + \
            f"--model {actor_model_path}" + " " + \
            f"--external_plugins ./examples/train/grpo/plugin/plugin.py" + " " + \
            f"--reward_funcs format" + " " + \
            f"--use_vllm false" + " " + \
            f"--vllm_gpu_memory_utilization {gpu_memory_utilization}" + " " + \
            f"--temperature {temperature}" + " " + \
            f"--top_p {top_p}" + " " + \
            f"--top_k {top_k}" + " " + \
            f"--train_type full" + " " + \
            f"--torch_dtype bfloat16" + " " + \
            f"--dataset {grpo_train_dataset_path_actor}" + " " + \
            f"--split_dataset_ratio 0" + " " + \
            f"--max_completion_length {max_len}" + " " + \
            f"--num_train_epochs 1" + " " + \
            f"--per_device_train_batch_size {per_device_train_batch_size}" + " " + \
            f"--per_device_eval_batch_size 1" + " " + \
            f"--learning_rate {lr}" + " " + \
            f"--warmup_ratio 0" + " " + \
            f"--lr_scheduler_type constant" + " " + \
            f"--gradient_accumulation_steps {gradient_accumulation_steps}" + " " + \
            f"--eval_strategy no" + " " + \
            f"--save_strategy epoch" + " " + \
            f"--save_total_limit 10" + " " + \
            f"--logging_steps 3" + " " + \
            f"--max_length {max_len}" + " " + \
            f"--output_dir {cur_iter_model_output_dir_actor}" + " " + \
            f"--dataloader_num_workers 4" + " " + \
            f"--dataset_num_proc 4" + " " + \
            f"--num_generations {num_generation}" + " " + \
            f"--repetition_penalty 1.05" + " " + \
            f"--system ./prompt.txt" + " " + \
            f"--deepspeed {deepspeed}" + " " + \
            f"--log_completions true" + " " + \
            f"--num_iterations 1" + " " + \
            f"--async_generate false" + " " + \
            f"--beta {kl_beta}" + " " + \
            f"--add_version False" + " " + \
            f"--model_type qwen2_5_vl" + " " + \
            f"--create_checkpoint_symlink True" + " " + \
            f"--freeze_vit True"
            # f"--max_grad_norm 1.0"

        grpo_command += grpo_args

        
        logger.info(grpo_command)
        # 生成stage1_sample.sh
        stage3_template_path = os.path.join(os.path.join(args.bash_path, 'template'), 'stage2_train_template.sh')
        stage3_train_path = os.path.join(args.bash_path, 'stage3_train.sh')
        with open(stage3_template_path, 'r', encoding='utf-8') as f:
            stage3_template = f.readlines()
        stage3_template.append(grpo_command)
        with open(stage3_train_path, 'w', encoding='utf-8') as f:
            f.writelines(stage3_template)
        stage3_train_submit_path = stage3_train_path.replace('.sh', '_submit.sh')


        # ==================================================================
        #          Submit: 3 训练任务提交(actor)
        # ==================================================================
        train_begin = False
        while not train_begin:
        
            # debug
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(5)
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(5)
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(5)
            
            os.system(f'bash {stage3_train_submit_path} {NGPUS} {NNODES} {args.task_name}')
            time.sleep(5)
            os.system('yes | ky exp delete -n train-h200-v000')
            time.sleep(30) # 等待提交成功


            os.system(occupy_shell)
            time.sleep(10)
            os.system(occupy_shell)
            
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GRPO 训练任务提交完成")
            # 监控是否生成了logging.jsonl
            logging_jsonl_path = os.path.join(cur_iter_model_output_dir_actor, 'logging.jsonl')
            print(f"[*] [{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始监控 {logging_jsonl_path}, 等待600s")
            time.sleep(600) # 等待20min, 60*20=1200
            if os.path.exists(logging_jsonl_path):
                train_begin = True
            else:
                train_begin = False # 没有启动训练, 再次提交
                # # kill训练任务
                train_task_name = f'train-grpo-sh-{args.task_name}'
                os.system(f'yes | ky exp delete -n {train_task_name}')
                time.sleep(5)
                os.system(f'yes | ky exp delete -n {train_task_name}')
                time.sleep(5)
                os.system(f'yes | ky exp delete -n {train_task_name}')

                print(f"[*] [{time.strftime('%Y-%m-%d %H:%M:%S')}] 没有启动GRPO训练, 再次重提")



        # 判断stage3_train是否结束, 监控model_output/iter_{i}/images
        monitor_folder(os.path.join(cur_iter_model_output_dir_actor, 'images'), duration_seconds=int(args.duration_seconds / 2), check_interval_seconds=args.check_interval_seconds)
        
        # ==================================================================
        #          Step: 3 GRPO训练完成(actor)
        # ==================================================================
        
        
        # 更新model_path
        actor_model_path = os.path.join(cur_iter_model_output_dir_actor, 'last')
        
        args.skip_first_sample = 'False'










