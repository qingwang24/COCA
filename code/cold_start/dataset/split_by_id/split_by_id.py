import os
import json
import multiprocessing
from collections import defaultdict
from tqdm import tqdm
import hashlib
def get_partition(question_id, num_partitions):
    """对字符串 question_id 进行哈希，确保适用于所有情况"""
    hash_value = int(hashlib.md5(question_id.encode()).hexdigest(), 16)  # 计算哈希
    return hash_value % num_partitions  # 取模后返回分区号
# 配置文件夹路径
data_folder = "/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v2/dataset/split_qwen2vl"  
output_folder = "/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_by_id/qwen2vl/rest/"  # 归类后保存的文件夹
# final_output_folder = "/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v2/dataset/split_by_id/qwen2.5vl/2"  # 最终合并后的文件夹

# **文件夹分区参数**
NUM_PARTITIONS = 1  # 题号按哈希值存入 10 个子目录，避免单个目录过载

# 创建子目录
for i in range(NUM_PARTITIONS):
    os.makedirs(os.path.join(output_folder, str(i)), exist_ok=True)

# 获取所有 JSON 文件
# json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
json_files = ['/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/exp/v2_1/multi_revision/qwen2vl_rest_neg/rest_neg.json',
            '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/ori/qwen2vl/rest/rest_pos_rest1.json',
            '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/ori/qwen2vl/rest/rest_pos.json'
                ]

# 共享锁字典 (防止多个进程同时写入同一题号)
manager = multiprocessing.Manager()
locks = manager.dict()


# 处理单个 JSON 文件
def process_file(filename):
    # file_path = os.path.join(data_folder, filename)
    file_path = filename
    local_data = defaultdict(lambda: {"pos_0": [], "pos": [], "neg": []})

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                return None  # 确保文件内容是列表

            for entry in tqdm(data):
                if "id" in entry:
                    question_id = entry["id"].split("_cri")[0]  # 提取题号
                    partition = get_partition(question_id, NUM_PARTITIONS)  # 计算存储目录

                    if "_cri_pos_0" in entry["id"] or "_cri_pos_rest_0" in entry["id"]:
                        local_data[question_id]["pos_0"].append(entry)
                    elif "_cri_pos" in entry["id"]:
                        local_data[question_id]["pos"].append(entry)
                    elif "_cri_neg" in entry["id"]:
                        if "revision" in entry and isinstance(entry["revision"], list) and len(entry["revision"]) == 10:
                            local_data[question_id]["neg"].append(entry)
        except json.JSONDecodeError:
            print(f"无法解析 JSON 文件: {filename}")
            return None

    # **按题号合并并存储**
    for q_id, categories in tqdm(local_data.items()):
        partition = get_partition(question_id, NUM_PARTITIONS)
        save_merged_file(q_id, categories, partition)


def save_merged_file(q_id, categories, partition):
    """ 使用进程锁确保不会多个进程同时写入同一个文件 """
    base_path = os.path.join(output_folder, str(partition), f"{q_id}.json")

    # 获取当前题号的锁 (如果不存在则创建)
    if q_id not in locks:
        locks[q_id] = manager.Lock()
    
    with locks[q_id]:  # 加锁
        # 读取旧数据，合并后写回
        if os.path.exists(base_path):
            with open(base_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    categories["pos_0"].extend(existing_data.get("pos_0", []))
                    categories["pos"].extend(existing_data.get("pos", []))
                    categories["neg"].extend(existing_data.get("neg", []))
                except json.JSONDecodeError:
                    print(f"文件损坏，无法解析: {base_path}")

        # 保存合并后的数据
        with open(base_path, "w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=4)


# 多进程处理文件
with multiprocessing.Pool(processes=3) as pool:
    list(tqdm(pool.imap(process_file, json_files), total=len(json_files), desc="Processing Files"))

print(f"所有数据已分类存储到 {output_folder} 目录！")