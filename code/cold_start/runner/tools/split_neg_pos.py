import json
import random
from tqdm import tqdm
import os
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

def process_data(input_path, output_path, positive=False, sample_size=1000):
    with open(input_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    sampled_data = random.sample(original_data, min(sample_size, len(original_data)))
    
    processed_data = []
    num_count = []
    pos_neg = []
    neg_data = []
    pos_data = []
    for item in tqdm(sampled_data):
        cur_data = []
        cur_pos_data = []
        cur_neg_data = []
        # if not 'm3cot' in item["id"].lower():
        #     continue
        # if'val' in item["id"].lower() and 'mmmu' not in item["id"].lower():
        #     continue
        try:
            # 提取原始字段
            preserved_fields = {
                "imgpath": item["imgpath"],
                # "id": item["id"],
                "target": item["target"]
            }
            id_name = item['id']
            
            # 处理prompt
            raw_prompt = item["prompt"]
            extracted_prompt = extract_prompt_content(raw_prompt)
            if not extracted_prompt:
                continue
            
            wrong_paths_not_rollout = [wrong_element for wrong_element in item["wrong_paths"] if wrong_element[3]==False]
            wrong_paths_rollout = [wrong_element for wrong_element in item["wrong_paths"] if wrong_element[3]==True]
            right_paths = item["right_paths"]

            random.shuffle(wrong_paths_not_rollout)
            random.shuffle(wrong_paths_rollout)
            random.shuffle(right_paths)
            # right_num = len(right_paths)
            select_label = item['select_label']
           
            prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'+extracted_prompt+'<|im_end|>\n<|im_start|>assistant\nLet\'s think step by step.'

            cur_pos_data.append({
                        **preserved_fields,
                        "id": id_name+f'_cri_pos_0',
                        "prompt": prompt,
                        "question": extracted_prompt.split('Question: ')[-1],
                        "query": extracted_prompt,
                        "label": select_label[0]+select_label[1],
                        "critique": '''{\n    "critique": ["Content is correct, format is correct, no corrections needed."]\n}''',
                        "cur_score": select_label[-1]
                    })

            max_num = 3 * min(len(right_paths), len(item["wrong_paths"]))
            max_num=10000000000
            if len(right_paths) >= len(item["wrong_paths"]):
                more_pos = True
            else:
                more_pos = False

            right_idx = 0
            for right_element in right_paths:
                if right_element[0]+right_element[1] == select_label[0]+select_label[1]:
                    continue
                if more_pos:
                    if len(cur_pos_data) >= max_num:
                        break
                right_idx+=1
                cur_pos_data.append({
                        **preserved_fields,
                        "id": id_name+f'_cri_pos_{right_idx}',
                        "prompt": prompt,
                        "question": extracted_prompt.split('Question: ')[-1],
                        "query": extracted_prompt,
                        "label": right_element[0]+right_element[1],
                        "critique": '''{\n    "critique": ["Content is correct, format is correct, no corrections needed."]\n}''',
                        "cur_score": right_element[-1]
                    })
            wrong_idx = -1
            for wrong_element in wrong_paths_not_rollout:
                if not more_pos:
                    if len(cur_neg_data) >= max_num:
                        break
                wrong_idx += 1
                cur_neg_data.append({
                            **preserved_fields,
                            "id": id_name+f'_cri_neg_{wrong_idx}',
                            "prompt": prompt,
                            "question": extracted_prompt.split('Question: ')[-1],
                            "query": extracted_prompt,
                            "selected_right": select_label,
                            "selected_wrong": wrong_element,
                            "rollout": False,
                        })
            for wrong_element in wrong_paths_rollout:
                if not more_pos:
                    if len(cur_neg_data) >= max_num:
                        break
                wrong_idx += 1
                cur_neg_data.append({
                            **preserved_fields,
                            "id": id_name+f'_cri_neg_{wrong_idx}',
                            "prompt": prompt,
                            "question": extracted_prompt.split('Question: ')[-1],
                            "query": extracted_prompt,
                            "selected_right": select_label,
                            "selected_wrong": wrong_element,
                            "rollout": True,
                        })
            processed_data += cur_neg_data + cur_pos_data
            neg_data+=cur_neg_data
            pos_data+=cur_pos_data
            num_count.append(len(cur_data))
            pos_neg.append([len(cur_pos_data), len(cur_neg_data)])
    
        except (KeyError, IndexError, TypeError) as e:
            print(f"数据异常跳过: {str(e)}")
            continue
    print(len(neg_data))
    print(len(pos_data))
    basename = os.path.basename(input_path).split('.json')[0]
    with open(output_path+basename+'_pos.json', 'w', encoding='utf-8') as f:
        json.dump(pos_data, f, ensure_ascii=False, indent=2)
    with open(output_path+basename+'_neg.json', 'w', encoding='utf-8') as f:
        json.dump(neg_data, f, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    process_data('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/ori/qwen2vl/choice_m3cot/choice_m3cot.json', '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/ori/qwen2vl/choice_m3cot/', False, 10000000000)
