import pickle
import math
import random
import re
import random
import numpy as np
from PIL import Image
def RA(gt, pred, ebsilon=0.05):
    try:
        gt = float(gt)
        pred = float(pred)
        if abs(gt - pred) / abs(gt) <= ebsilon:
            return 1 - abs(gt - pred) / abs(gt)
        else:
            return 0.0
    except:
        return float(str(gt).lower().replace(' ','') == str(pred).lower().replace(' ',''))


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, question, target, numMCTSSims, cpuct, expand_num, imgpath, pixel_values, thw):
        self.root = question # 指示根节点的字符串
        self.target = target # 指示目标答案b 
        self.imgpath = imgpath
        self.pixel_values = pixel_values
        self.thw = thw

        self.numMCTSSims = numMCTSSims
        self.cpuct = cpuct
        self.expand_num = expand_num

        self.Qsa = {} # 指示当前节点 s 执行 a 估计的 Value
        self.Qsa_list = {} # 指示当前节点 s 执行 a 估计的 Value_list
        self.Qsa_rollout_list = {} # 指示当前节点 s 执行 a 估计的 Value_list
        self.Nsa = {} # 指示当前节点 s 执行 a 的次数
        self.Ns = {}  # 指示当前节点 s 的访问次数
        self.Ps = {}  # 指示当前节点 s 可选动作的概率分布 (依据policy model 似然概率估计)
        self.Es = {}  # 指示当前节点 s 是否结束

    def select_node(self, cul_steps):
        '''
        依据当前状态,挑选节点
        
        Returns:
            is_end: True/False 指示是否是终端节点
            select_cul_steps: string 指示选中的解码步骤
        '''
        s = cul_steps
        if s not in self.Es:
            self.Es[s] = self.getStepEnded(s) # 当前解码步骤是否结束
        if self.Es[s]:
            # terminal node
            return True, s
        
        if s not in self.Ps: # is not explore
            return False, s
        
        if s in self.Ps:
            if 'next_step_proposals' not in self.Ps[s]: # is not fully expand
                return False, s

        cur_best = -float('inf')
        best_act = ''

        # select: pick the action with the highest upper confidence bound
        for action_index, a in enumerate(self.Ps[s]['next_step_proposals']):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s]['next_step_proposals_probs'][action_index] \
                    * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s,a)])
            else:
                u = self.cpuct * self.Ps[s]['next_step_proposals_probs'][action_index] \
                    * math.sqrt(self.Ns[s] + 10e-8)

            if u > cur_best:
                cur_best = u
                best_act = a

        next_s = self.get_next_state(s, best_act)

        # find leaf node
        result = self.select_node(next_s)
        return result
    
    def update_expand_node(self, cul_steps, next_step_proposals, next_step_proposals_probs):
        '''
        将选中节点expand的结果进行更新

        Return:
            None
        '''
        s = cul_steps
        if s not in self.Ps:
            self.Ps[s] = dict()
            self.Ps[s]['next_step_proposals'] = next_step_proposals
            self.Ps[s]['next_step_proposals_probs'] = next_step_proposals_probs
        else:
            if 'next_step_proposals' not in self.Ps[s]:
                self.Ps[s]['next_step_proposals'] = next_step_proposals
                self.Ps[s]['next_step_proposals_probs'] = next_step_proposals_probs
            else:
                print("error process update expand node information")
                import pdb
                pdb.set_trace()

        for a in next_step_proposals:
            next_s = self.get_next_state(s, a)
            self.Ps[next_s] = dict()
            self.Ps[next_s]['parent'] = s
            self.Ps[next_s]['action'] = a

        self.Ns[s] = 0
        return None

    def select_expand_node(self, cul_steps):
        '''
        依据UCT公式,从当前节点挑选合适的叶子节点

        Return:
            select_node: 选中节点内容
        '''

        cur_best = -float('inf')
        best_act = ''
        s = cul_steps
        # select: pick the action with the highest upper confidence bound
        for action_index, a in enumerate(self.Ps[s]['next_step_proposals']):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s]['next_step_proposals_probs'][action_index] \
                    * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s,a)])
            else:
                u = self.cpuct * self.Ps[s]['next_step_proposals_probs'][action_index] \
                    * math.sqrt(self.Ns[s] + 10e-8)

            if u > cur_best:
                cur_best = u
                best_act = a

        return best_act
    
    def backpropagate(self, cul_steps, action, value, value_list, rollout_list):
        # backpropagate
        s = cul_steps
        a = action
        v = value
        # assert len(value_list)==3 or len(value_list)==1
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
            self.Qsa_list[(s, a)].append(value_list)
            self.Qsa_rollout_list[(s, a)].append(rollout_list)
        else:
            self.Ps[self.get_next_state(s,a)]['value'] = v # store origin value
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
            self.Qsa_list[(s, a)] = [value_list]
            self.Qsa_rollout_list[(s, a)] = [rollout_list]

        self.Ns[s] += 1

        if 'parent' in self.Ps[s] and "action" in self.Ps[s]:
            state, action = self.Ps[s]['parent'], self.Ps[s]['action']
            self.backpropagate(state, action, v, value_list, rollout_list)

    def getStepEnded(self, cul_steps):
        """
        获取解码到当前步骤是否已经解码结束.

        Returns:
            flag: True/False
        """
        tmp = cul_steps.strip()
        if tmp.count("<|im_end|>")==3  and not tmp.endswith("<|im_end|>"):
            print(tmp.replace('<|image_pad|>',''))
            print(self.imgpath)
        if tmp.endswith("<|im_end|>"): # TODO for debug
            return True
        else:
            return False

    def get_next_state(self, cul_steps, select_action):
        next_state = cul_steps + select_action
        return next_state
    
    def get_value(self, cul_step):
        state = cul_step
        parent, action = self.Ps[state]['parent'], self.Ps[state]['action']
        return self.Qsa[(parent, action)]

import os
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
HF_DICT = {
    "llava-hf/llava-v1.6-mistral-7b-hf": "/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/llava-v1.6-mistral-7b-hf",
    "Qwen/Qwen2-VL-2B-Instruct": "/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-2B-Instruct",
    "openbmb/MiniCPM-V-2": "/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/MiniCPM-V-2"
    }
pretrained_path = HF_DICT["Qwen/Qwen2-VL-2B-Instruct"]
tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
# files = os.listdir('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/train_fillin/expand11_step30_Sims1_rollout16_t0.7_p0.9_k50/mcts_tree/')
select_files = []
# random.shuffle(files)
size_list=[]
size1_list=[]
imgsize_list=[]
imgthw_list=[]
token_num=[]
token_num_=[]
err_num=0

import os
import pickle
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
# false_id_list = json.load(open('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/BoN/test_mcts/qwen2vl/BoN_1_202502251050/split/false_id_list.json'))
def process_file(file_path):
    # if os.path.basename(file_path).split('.')[0] not in false_id_list:
    #     return None
    try:
        with open(file_path, "rb") as f:
            mcts_tree = pickle.load(f)
    except:
        # err_num+=1
        return None, None

    id_name = mcts_tree.id.lower()
    # if 'mmmu_val' in id_name and 'choice' not in id_name:
        # return None, None
    # if 'm3cot' in id_name and 'choice' not in id_name:
    #     return None, None
    layer_num = len(mcts_tree.Es)
    layer = 0
    Qsa_dict = mcts_tree.Qsa
    Qsa = list(Qsa_dict.items())
    Qsa_list_dict = mcts_tree.Qsa_list
    Qsa_rollout_list_dict = mcts_tree.Qsa_rollout_list
    Qsa_list = list(Qsa_list_dict.items())
    Ps_dict = mcts_tree.Ps
    Ps = list(Ps_dict.items())
    target=mcts_tree.target
    root = mcts_tree.root
    pattern = r"<\|im_start\|>user(.*?)<\|im_end\|>"
    question = re.findall(pattern, root, re.DOTALL)[0].split('<|vision_end|>')[-1].split('Hint:')[-1].strip().split('\nQuestion:')[-1]
    node = root
    steps_list = []
    wrong_paths=[]
    right_paths=[]

    wrong_answers = []
    right_answers = []
    labels = []
    select_label = None
    for node in mcts_tree.Es.keys():
        layer+=1
        node_Ps = Ps_dict[node]
        part_answer = node.split('<|im_start|>assistant\n')[-1]
        if 'next_step_proposals' in node_Ps.keys():
            next_step_proposals = node_Ps['next_step_proposals']
            value_list = []
            value_list1 = []
            strict_value_list = []
            for child in next_step_proposals:
                v_list = Qsa_list_dict[(node, child)][0]
                strict_list = [1 if v > 0.8 else 0 for v in v_list]
                strict_value = np.mean(strict_list)
                v_list_ = [1 if v > 0.5 else 0 for v in v_list]
                value = np.mean(v_list_)
                if child.endswith('<|im_end|>'):
                    sa = part_answer + child
                    assert sa[:25] == '''Let's think step by step.'''
                    sa=sa[25:]
                    if value==0:
                        if part_answer+child.replace('<|im_end|>','') not in wrong_answers:
                            # if sa.replace('<|im_end|>','').replace(' ','')!='':
                            wrong_paths.append([part_answer, child, layer, False, np.mean(v_list)])
                            wrong_answers.append(part_answer+child.replace('<|im_end|>',''))
                    elif value==1: #and layer_num==layer:
                        if sa not in labels:
                            right_paths.append([part_answer, child, layer, False, np.mean(v_list)]) #是否为rollout结果
                            right_answers.append(part_answer+child.replace('<|im_end|>',''))
                            labels.append(sa)
                value_list.append(value)
                value_list1.append(Qsa_dict[(node, child)])
                strict_value_list.append(strict_value)
            max_value = max(value_list)
            strict_max_value = max(strict_value_list)
            min_value = min(value_list)
            if layer_num==layer:#最后一层
                max_child = next_step_proposals[strict_value_list.index(strict_max_value)]
                if max_child.endswith('<|im_end|>') and strict_max_value>0.8:
                    assert np.mean(Qsa_list_dict[(node, max_child)][0])>0.8
                    select_label = [part_answer, max_child, layer, False, np.mean(Qsa_list_dict[(node, max_child)][0])] #mcts选择的答案

            for child_id, child_value in enumerate(value_list):
                if child_value < 1:
                    # for min_child in next_step_proposals:
                    min_child = next_step_proposals[child_id]
                    if not min_child.endswith('<|im_end|>'):
                        v_list = Qsa_list_dict[(node, min_child)][0]
                        r_list = Qsa_rollout_list_dict[(node, min_child)][0]
                        for rollout_child, rollout_value in zip(r_list, v_list):
                            if rollout_value < 0.5:
                                rollout_child = rollout_child.split('<|im_end|>\n<|im_start|>assistant\n')[-1]
                                rollout_child = rollout_child[len(part_answer):]
                                if part_answer + rollout_child.replace('<|im_end|>','') not in wrong_answers:
                                    # if sa.replace('<|im_end|>','').replace(' ','')!='':
                                    wrong_paths.append([part_answer, rollout_child, layer, True, rollout_value])
                                    wrong_answers.append(part_answer+rollout_child.replace('<|im_end|>',''))
                                    break
                if child_value > 0:
                    # for min_child in next_step_proposals:
                    min_child = next_step_proposals[child_id]
                    if not min_child.endswith('<|im_end|>'):
                        v_list = Qsa_list_dict[(node, min_child)][0]
                        r_list = Qsa_rollout_list_dict[(node, min_child)][0]
                        for rollout_child, rollout_value in zip(r_list, v_list):
                            if rollout_value > 0.5:
                                rollout_child = rollout_child.split('<|im_end|>\n<|im_start|>assistant\n')[-1]
                                rollout_child = rollout_child[len(part_answer):]
                                if part_answer + rollout_child.replace('<|im_end|>','') not in right_answers:
                                    # if sa.replace('<|im_end|>','').replace(' ','')!='':
                                    right_paths.append([part_answer, rollout_child, layer, True, rollout_value])
                                    right_answers.append(part_answer+rollout_child.replace('<|im_end|>',''))
                                    break


    if len(right_paths)>0 and len(wrong_paths)>0:
        if select_label is None:
            select_label=right_paths[-1]
    # if len(right_paths)>0:
        return dict(id=mcts_tree.id, imgpath=mcts_tree.imgpath, target=target, prompt=root, select_label=select_label, labels=labels, right_paths=right_paths, thw=mcts_tree.thw, wrong_paths=wrong_paths, layer_num=layer_num), True

    elif len(right_paths)>0 and len(wrong_paths)==0:
        if select_label is None:
            select_label=right_paths[-1]
    # if len(right_paths)>0:
        return dict(id=mcts_tree.id, imgpath=mcts_tree.imgpath, target=target, prompt=root, select_label=select_label, labels=labels, right_paths=right_paths, thw=mcts_tree.thw, wrong_paths=wrong_paths, layer_num=layer_num), False
    else:
        return 1, 1

def main():

    err_num=0
    train_choice_truefalse_2_5_path = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/train_choice_truefalse_2_5/expand6_step30_Sims1_rollout10_t0.7_p0.9_k50/mcts_tree'
    train_chartqa_m3cot_2_5_path = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/train_chqrtqa_m3cot_2_5/expand11_step30_Sims1_rollout16_t0.7_p0.9_k50/mcts_tree'
    train_fillin_2_5_path = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/train_fillin_2_5/expand6_step30_Sims1_rollout10_t0.7_p0.9_k50/mcts_tree'
    test_val_all_2_5_path = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/test_val_all_2_5/expand11_step30_Sims1_rollout16_t0.7_p0.9_k50/mcts_tree'
    base_paths = [train_chartqa_m3cot_2_5_path,
                    train_fillin_2_5_path,
                    train_choice_truefalse_2_5_path]

    base_paths = ['/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/train_chqrtqa_m3cot_2_5_new/expand11_step30_Sims1_rollout16_t0.7_p0.9_k50/mcts_tree']

    # base_paths = [train_choice_truefalse_2_5_path, train_chartqa_m3cot_2_5_path]
    # base_paths = [test_val_all_2_5_path]
    # base_paths = [train_chartqa_m3cot_2_5_path]
    # base_paths = ['/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/mcts_alpha/exp/train_fillin/expand11_step30_Sims1_rollout16_t0.7_p0.9_k50/mcts_tree_1']
    file_paths = []
    files_all = []
    for base_path in base_paths:
        files = os.listdir(base_path)
        # if 'train_fillin_2_5' in base_path:
        #     # files = [f for f in files if 'train_aug' not in f and 'train_human' not in f]
        #     files = [f for f in files if f not in files_all]
        # if 'train_chqrtqa_m3cot_2_5' in base_path:
        #     files = [f for f in files if 'm3cot' in f.lower()]

        # random.shuffle(files)
        file_paths += [os.path.join(base_path, file) for file in files]
        files_all += files
    random.shuffle(file_paths)
    neg_list = []
    pos_list = []
    err_num=0
    with Pool(40) as pool:
        for size, flag in tqdm(pool.imap(process_file, file_paths), total=len(file_paths)):
            if size is not None and size!=1 and flag is True:
                neg_list.append(size)
            if size is not None and size!=1 and flag is False:
                pos_list.append(size)
            if size is None:
                err_num+=1
    print(len(neg_list))
    print(len(pos_list))
    with open('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/ori/qwen2.5vl/m3cot_chartqa/m3cot_chartqa.json', 'w') as f:
        json.dump(neg_list, f)
    with open('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/ori/qwen2.5vl/m3cot_chartqa/m3cot_chartqa_pos_rest.json', 'w') as f:
        json.dump(pos_list, f)


main()

