import json
import random
from tqdm import tqdm
import os
import multiprocessing

import os
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, Manager


# def main():
#     # folder_path = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v2/dataset/split_by_id/qwen2.5vl/0'  # 替换为实际路径
#     # files = os.listdir(folder_path)
#     pass_rate=dict()
#     random.shuffle(file_paths)
#     for file in tqdm(file_paths):
#         with open(file) as f:
#             da = json.load(f)
#         neg_list = da['neg']
#         if len(neg_list)==0:
#             continue
#         max_layer = neg_list[0]['selected_right'][2]
#         max_layer1 = max([sample['selected_wrong'][2] for sample in neg_list])
#         # if max_layer!=3:
#         #     continue
#         if max_layer1>max_layer:
#             continue

#         for sample in neg_list:
#             if sample['rollout']:
#                 continue
#             layer = sample['selected_wrong'][2]
#             if layer not in pass_rate.keys():
#                 pass_rate[layer] = dict(total=0, passed=0)
#             else:
#                 pass_rate[layer]['total']+=1
#                 if sum(1 for score in sample['pigai_score'] if score > 0.5) > 3:
#                     pass_rate[layer]['passed']+=1
#         1
#     1
        

            

        
# if __name__ == "__main__":
#     main()
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from glob import glob

thresholds = [2, 3, 4, 5, 6, 7]

def process_file(file):
    with open(file) as f:
        da = json.load(f)

    neg_list = da.get('neg', [])
    if len(neg_list) == 0:
        return [{},[]]

    max_layer = da['neg'][0]['selected_right'][2]
    max_layer1 = max(sample['selected_wrong'][2] for sample in neg_list)
    # if max_layer1 > max_layer:
    #     return {}

    local_pass_rate = {}

    pass_samples = []
    for sample in neg_list:
        # if sample.get('rollout'):
        #     continue
        layer = sample['selected_wrong'][2]

        layer = min(max_layer, layer)

        if layer not in local_pass_rate:
            local_pass_rate[layer] = {t: {'total': 0, 'passed': 0} for t in thresholds}

        pigai_score = sample.get('pigai_score', [])
        count = sum(1 for score in pigai_score if score > 0.5)

        if count > 3 and count < 5:
            pass_samples.append(sample)

        for t in thresholds:
            local_pass_rate[layer][t]['total'] += 1
            if count > t:
                local_pass_rate[layer][t]['passed'] += 1

    return [local_pass_rate, pass_samples]


def merge_pass_rate(all_results):
    merged = {}
    all_samples = []
    for res in all_results:
        if res[1] != []:
            all_samples.append(res[1])
        res = res[0]
        for layer, stats_per_threshold in res.items():
            if layer not in merged:
                merged[layer] = {t: {'total': 0, 'passed': 0} for t in thresholds}
            for t in thresholds:
                merged[layer][t]['total'] += stats_per_threshold[t]['total']
                merged[layer][t]['passed'] += stats_per_threshold[t]['passed']
    return merged, all_samples


if __name__ == "__main__":
    # file_paths = glob("your_path/*.json")  # 替换成你自己的路径
    folder_paths = ['/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_by_id/qwen2vl/choice_m3cot/0',
                '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_by_id/qwen2vl/mathv_chartqa/0',
                '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_by_id/qwen2vl/rest/0']
    file_paths = []
    for folder_path in folder_paths:
        files=os.listdir(folder_path)
        file_paths+=[os.path.join(folder_path, f) for f in files]
    save_root = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/dataset/split_for_pack/qwen2vl'
    random.shuffle(file_paths)
    MAX_PER_FILE = 100000
    random.shuffle(file_paths)

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, file_paths), total=len(file_paths)))

    pass_rate, all_samples = merge_pass_rate(results)

    # 输出
    for layer in sorted(pass_rate.keys()):
        print(f"Layer {layer}:")
        for t in thresholds:
            total = pass_rate[layer][t]['total']
            passed = pass_rate[layer][t]['passed']
            rate = passed / total if total > 0 else 0
            print(f"  Threshold >{t}: {passed}/{total} ({rate:.2%})")


1