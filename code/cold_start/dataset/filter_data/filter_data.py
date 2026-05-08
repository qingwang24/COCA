import json
import random
from tqdm import tqdm
import os
import multiprocessing
folder_paths = ['/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_by_id/qwen2vl/choice_m3cot/0',
                '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_by_id/qwen2vl/mathv_chartqa/0',
                '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_by_id/qwen2vl/rest/0']
file_paths = []
for folder_path in folder_paths:
    files=os.listdir(folder_path)
    file_paths+=[os.path.join(folder_path, f) for f in files]
save_root = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2vl'
MAX_PER_FILE = 100000
import os
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, Manager

def process_file(file):
    with open(file) as f:
        da = json.load(f)
    
    pos_0 = da['pos_0']
    pos_list = da['pos']
    neg_list = da['neg']

    
    neg_list = [sample for sample in neg_list if sum(1 for score in sample['pigai_score'] if score > 0.5) > 3]
    num_neg = len(neg_list)

    if num_neg > 0 and len(pos_0) == 0:
        neg_sample = neg_list[0]
        id_name = neg_sample['id'].split('_cri')[0]
        preserved_fields = {k: neg_sample[k] for k in ["imgpath", "target", "prompt", "question", "query"]}
        pos_0=[]
        pos_0.append({
                        **preserved_fields,
                        "id": f"{id_name}_cri_pos_0",
                "label": neg_sample['selected_right'][0] + neg_sample['selected_right'][1],
                "critique": '''{"critique": ["Content is correct, format is correct, no corrections needed."]}''',
                "cur_score": neg_sample['selected_right'][-1]
                    })

    num_pos = len(pos_list) + len(pos_0)
    if num_pos == 0:
        return None
    
    if num_neg == 0:
        select_neg = []
        select_pos = pos_0 + random.sample(pos_list, min(len(pos_list), 2))
    elif num_pos / num_neg > 3:
        select_num = num_neg * 3
        select_neg = neg_list
        select_pos = pos_0 + random.sample(pos_list, min(len(pos_list), select_num - 1))
    elif num_neg / num_pos > 3:
        revised_list = []
        select_num = int(num_neg / 3)
        for neg_sample in neg_list:
            preserved_fields = {k: neg_sample[k] for k in ["imgpath", "target", "prompt", "question", "query"]}
            id_name = neg_sample['id']
            right_idx = -1
            for revised_sample, revised_score in zip(neg_sample['revision'], neg_sample['pigai_score']):
                if revised_score > 0.8:
                    right_idx += 1
                    revised_list.append({
                        **preserved_fields,
                        "id": f"{id_name}_revised_{right_idx}",
                        "label": "Let's think step by step." + revised_sample,
                        "critique": '''{"critique": ["Content is correct, format is correct, no corrections needed."]}''',
                        "cur_score": revised_score
                    })
        select_pos = pos_0 + pos_list + random.sample(revised_list, min(len(revised_list), select_num - num_pos))
        select_neg = neg_list
    else:
        select_pos = pos_0 + pos_list
        select_neg = neg_list
    
    if 'mathv360k' in file.lower():
        return ('mathv', select_neg, select_pos)
    elif 'm3cot' in file.lower():
        return ('m3cot', select_neg, select_pos)
    else:
        return ('chartqa', select_neg, select_pos)
def save_category_data(category, data_list):
    """保存一个类别的数据，并拆分文件"""
    category_path = os.path.join(save_root, category)
    os.makedirs(category_path, exist_ok=True)  # 确保文件夹存在

    total = len(data_list)
    num_files = (total // MAX_PER_FILE) + (1 if total % MAX_PER_FILE else 0)  # 计算需要拆分的文件数

    for i in range(num_files):
        start_idx = i * MAX_PER_FILE
        end_idx = min((i + 1) * MAX_PER_FILE, total)
        part_data = data_list[start_idx:end_idx]

        save_path = os.path.join(category_path, f"{category}_part{i+1}.json")  # 生成文件名

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(part_data, f, ensure_ascii=False, indent=4)

        print(f"✅ Saved {len(part_data)} samples to {save_path}")
def process_wrapper(f):
    return process_file(f)
def main():
    # folder_path = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v2/dataset/split_by_id/qwen2.5vl/0'  # 替换为实际路径
    # files = os.listdir(folder_path)
    
    with Manager() as manager:
        pass_rate = manager.dict()
        select_mathv_neg = manager.list()
        select_mathv_pos = manager.list()
        select_chartqa_neg = manager.list()
        select_chartqa_pos = manager.list()
        select_m3cot_neg = manager.list()
        select_m3cot_pos = manager.list()
        
        with Pool(processes=40) as pool:  # 你可以调整进程数
            results = list(tqdm(pool.imap(process_wrapper, file_paths), total=len(file_paths)))
            
        for result in tqdm(results):
            if result:
                category, neg, pos = result
                if category == 'mathv':
                    select_mathv_neg.extend(neg)
                    select_mathv_pos.extend(pos)
                elif category == 'm3cot':
                    select_m3cot_neg.extend(neg)
                    select_m3cot_pos.extend(pos)
                else:
                    select_chartqa_neg.extend(neg)
                    select_chartqa_pos.extend(pos)
        data_categories = {
        "mathv_neg": select_mathv_neg,
        "mathv_pos": select_mathv_pos,
        "chartqa_neg": select_chartqa_neg,
        "chartqa_pos": select_chartqa_pos,
        "m3cot_neg": select_m3cot_neg,
        "m3cot_pos": select_m3cot_pos,
    }   
        total_num=0
        for data_category, datas in data_categories.items():
            print(f"{data_category}: {len(datas)}")
            total_num+=len(datas)
        print(f"total_num: {total_num}")
        # 每个文件最多 100000 条数据
        MAX_PER_FILE = 100000
        # save_root = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2.5vl'  # 根目录

        os.makedirs(save_root, exist_ok=True)

        with multiprocessing.Pool(processes=8) as pool:
            pool.starmap(save_category_data, data_categories.items())
        
        1    
    # 你可以在这里保存 select_* 变量的数据

if __name__ == "__main__":
    main()
1