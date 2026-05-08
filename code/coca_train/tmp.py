import json
import os
import sys


input_json_path_list = [
    "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/RL_dataset_part_1/piagi_[tmp_total]_2025041818/BoN_all_result_total.json",
    "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/RL_dataset_part_2/piagi_[BoN_all_result_total]_2025041818/BoN_all_result_total.json",
    "/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/RL_dataset_part_3/piagi_[tmp_total]_2025041817/BoN_all_result_total.json"
]


total_dataset = []
for json_path in input_json_path_list:
    print(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        cur_dataset = json.load(f)
    total_dataset += cur_dataset


print(f"total dataset_length: {len(total_dataset)}")
output_path = '/train21/cog8/permanent/qkchang/R1_Zero/experiments/pigai_result/RL_dataset/RL_all_dataset_bon_6.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(total_dataset, f, indent=4, ensure_ascii=False)


