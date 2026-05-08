import json
import random
from tqdm import tqdm

def split_list(lst, n=4):
    return [lst[i:i + n] for i in range(0, len(lst), n)]
def extract_prompt_content(prompt_str):
    """提取<|vision_end|>和<|im_end|>之间的内容"""
    start_tag = "<|vision_end|>"
    end_tag = "<|im_end|>"
    
    start_idx = prompt_str.find(start_tag)
    if start_idx == -1:
        return None
    
    start_idx += len(start_tag)
    end_idx = prompt_str.find(end_tag, start_idx)
    
    if end_idx == -1:
        return None
    
    return prompt_str[start_idx:end_idx].strip()
max_rest_num=100000000

# train_all_pos = json.load(open('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/rest/train_all_pos.json'))


# train_all_pos_list = split_list(train_all_pos, 1600000)
# for split_id, train_all_pos in enumerate(train_all_pos_list):
#     sample_pos = []
#     for pos_data in tqdm(train_all_pos):
#         pos_data['cur_score'] = pos_data['select_label'][-1]
#         sample_pos.append(pos_data)
#     with open(f'/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/for_pack/qwen2.5vl/train_all_pos{split_id}.json', 'w', encoding='utf-8') as f:
#         json.dump(sample_pos, f, ensure_ascii=False, indent=2)
    



train_all_pos_rest = json.load(open('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/ori/qwen2vl/choice_m3cot/choice_m3cot_pos_rest.json'))
sample_rest_data_all = []
for rest_data in tqdm(train_all_pos_rest):
    right_paths = rest_data["right_paths"]
    random.shuffle(right_paths)
    select_label = rest_data['select_label']
    raw_prompt = rest_data["prompt"]
    extracted_prompt = extract_prompt_content(raw_prompt)           
    prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'+extracted_prompt+'<|im_end|>\n<|im_start|>assistant\nLet\'s think step by step.'
    preserved_fields = {
            "imgpath": rest_data["imgpath"],
            "target": rest_data["target"]
        }
    id_name = rest_data['id']
    right_idx = 0
    sample_rest_data_all.append({
                    **preserved_fields,
                    "id": id_name+f'_cri_pos_rest_0',
                    "prompt": prompt,
                    "question": extracted_prompt.split('Question: ')[-1],
                    "query": extracted_prompt,
                    "label": select_label[0]+select_label[1],
                    "critique": '''{\n    "critique": ["Content is correct, format is correct, no corrections needed."]\n}''',
                    "cur_score": select_label[-1]
                })
    if len(sample_rest_data_all)>=max_rest_num:
        break

    for right_element in right_paths:
        if right_element[0]+right_element[1] == select_label[0]+select_label[1]:
            continue
        if right_idx>=100 or len(sample_rest_data_all)>=max_rest_num:
            break
        right_idx+=1
        sample_rest_data_all.append({
                **preserved_fields,
                "id": id_name+f'_cri_pos_rest_{right_idx}',
                "prompt": prompt,
                "question": extracted_prompt.split('Question: ')[-1],
                "query": extracted_prompt,
                "label": right_element[0]+right_element[1],
                "critique": '''{\n    "critique": ["Content is correct, format is correct, no corrections needed."]\n}''',
                "cur_score": right_element[-1]
            })
        
    if len(sample_rest_data_all)>=max_rest_num:
        break
with open(f'/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/ori/qwen2vl/choice_m3cot/choice_m3cot_pos_rest1.json', 'w', encoding='utf-8') as f:
    json.dump(sample_rest_data_all, f, ensure_ascii=False, indent=2) 