import json
from tqdm import tqdm
import random

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

def map_score(score):
    assert score>=0 and score<=1
    if score < 0.3:
        score = 1
    elif score < 0.5:
        score = 2
    elif score < 0.7:
        score = 3
    elif score < 0.9:
        score = 4
    else:
        score = 5
    return score

datatype='m3cot'
modeltype='qwen2.5vl'


pos_data_list = json.load(open(f'/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/{modeltype}/train_{datatype}_pos.json'))
neg_data_list = json.load(open(f'/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/exp/v2_1/critique_revision/train_m3cot/train_{datatype}_neg.json'))
rest_data_list = json.load(open(f'/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/{modeltype}/train_{datatype}_pos_rest.json'))

data_dict=dict()
sample_rest_data_all = []
for pos_data in tqdm(pos_data_list):
    id_name = pos_data['id']
    base_id = id_name.split('_cri_')[0]
    # pos_data['cur_score'] = pos_data['select_label'][-1]
    if base_id in data_dict.keys():
        data_dict[base_id]['pos_data'].append(pos_data)
    else:
        data_dict[base_id] = dict(pos_data=[pos_data], neg_data=[])


for neg_data in tqdm(neg_data_list):
    # if neg_data['pigai_score']<=0.8:
    #     continue

    # if not (neg_data['pigai_score']>0.5 and neg_data['pigai_score']-neg_data['selected_wrong'][-1]>0.01):
    # if not (neg_data['pigai_score']-neg_data['selected_wrong'][-1]>0.01):
    #     continue


    id_name = neg_data['id']
    base_id = id_name.split('_cri_')[0]
    neg_data['cur_score'] = neg_data['selected_wrong'][-1]
    # revise_data = neg_data.copy()
    # revise_data['id'] = revise_data['id']+'_revised'
    # revise_data['label'] = '''Let's think step by step.'''+revise_data['revision']
    # revise_data['critique'] = '''{\n    "critique": ["Content is correct, format is correct, no corrections needed."]\n}'''
    # revise_data['cur_score'] = revise_data['pigai_score']
    if base_id in data_dict.keys():
        data_dict[base_id]['neg_data'].append(neg_data)
        # if revise_data['pigai_score'] > 0.5:
        #     data_dict[base_id]['pos_data'].append(revise_data)
    else:
        data_dict[base_id] = dict(pos_data=[], neg_data=[neg_data]) 
        # if revise_data['pigai_score'] > 0.5:
        #     data_dict[base_id] = dict(pos_data=[revise_data], neg_data=[]) 

sample_pos_data_all = []
sample_neg_data_all = []
ratio = []
pos_neg = []
for base_id, cri_data in tqdm(data_dict.items()):
    pos_datas = cri_data['pos_data']
    neg_datas = cri_data['neg_data']
    assert 'cri_pos_0' in pos_datas[0]['id']

    neg_datas_rollout = [neg_data for neg_data in neg_datas if neg_data['rollout']==True]
    neg_datas_not_rollout = [neg_data for neg_data in neg_datas if neg_data['rollout']==False]

    max_num = min(len(pos_datas), len(neg_datas)) * 3
    max_num = 1000000
    if len(neg_datas)==0:
        if len(pos_datas) > 1:
            sample_rest_data_all+=pos_datas[:1] + random.sample(pos_datas[1:], min(3, len(pos_datas[1:])))
        else:
            sample_rest_data_all+=pos_datas
        continue
    if len(pos_datas) > 1:
        sample_pos_data = pos_datas[:1] + random.sample(pos_datas[1:], min(max_num-1, len(pos_datas[1:])))
    else:
        sample_pos_data = pos_datas

    not_rollout_num = len(neg_datas_not_rollout)
    if not_rollout_num >= max_num:
        sample_neg_data = random.sample(neg_datas_not_rollout[:], max_num)
    else:
        sample_neg_data = neg_datas_not_rollout + random.sample(neg_datas_rollout[:], min(max_num-not_rollout_num, len(neg_datas_rollout)))

    sample_pos_data_all += sample_pos_data
    sample_neg_data_all += sample_neg_data
    ratio.append(len(sample_pos_data)/len(sample_neg_data))
    pos_neg.append([len(sample_pos_data),len(sample_neg_data)])

max_rest_num = len(sample_neg_data_all) - len(sample_pos_data_all)
max_rest_num = 10000000000
rest_count = []
if max_rest_num > 0:
    random.shuffle(rest_data_list)
    for rest_data in tqdm(rest_data_list):
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
            rest_count.append(right_idx+1)
            break

        for right_element in right_paths:
            if right_element[0]+right_element[1] == select_label[0]+select_label[1]:
                continue
            if right_idx>=0 or len(sample_rest_data_all)>=max_rest_num:
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
            rest_count.append(right_idx+1)
            break
        rest_count.append(right_idx+1)
# if len(sample_rest_data_all) > 0 and max_rest_num > 0:

print(len(sample_rest_data_all))
print(len(sample_pos_data_all))
print(len(sample_neg_data_all))

with open(f'/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/for_pack/{modeltype}/train_{datatype}_pos_rest.json', 'w', encoding='utf-8') as f:
    json.dump(sample_rest_data_all, f, ensure_ascii=False, indent=2)            


with open(f'/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/for_pack/{modeltype}/train_{datatype}_pos.json', 'w', encoding='utf-8') as f:
    json.dump(sample_pos_data_all, f, ensure_ascii=False, indent=2)

with open(f'/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/dataset/v2_1/for_pack/{modeltype}/train_{datatype}_neg.json', 'w', encoding='utf-8') as f:
    json.dump(sample_neg_data_all, f, ensure_ascii=False, indent=2)



