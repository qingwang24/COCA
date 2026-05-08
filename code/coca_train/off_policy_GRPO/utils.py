import os
import sys
from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoProcessor, Qwen2ForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
import torch
import re
from vllm import SamplingParams
sys.path.append('./')
sys.path.append('/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/')
from vision_processor import process_vision_info
import numpy as np
from torch.nn.utils.rnn import pad_sequence
def split_list(lst, n=4):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def expand_list(input_list, times=3):
    return [item for item in input_list for _ in range(times)]
class PigaiModel(Qwen2ForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.score = nn.Linear(config.hidden_size, 1)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            sigmoid_logits = torch.sigmoid(logits).squeeze(-1)
            loss = torch.sum(((sigmoid_logits - labels) ** 2 ) * loss_mask) / loss_mask.sum().clamp(min=1)
            if torch.isnan(loss):
                print('loss == nan detect! ')

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PigaiAccuracyORM():
    '''使用批改模型来提供reward'''

    def __init__(self, actor, actor_processor, device):
        # 加载pigai_processor 和 pigai_model
        self.pigai_processor = AutoProcessor.from_pretrained("/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/models/Qwen2.5-1.5B-Instruct")
        self.pigai_tokenizer = AutoTokenizer.from_pretrained("/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/models/Qwen2.5-1.5B-Instruct")
        self.pigai_model = PigaiModel.from_pretrained("/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/models/pigai_model/checkpoint-467/").to(device)
        self.pigai_model.eval()
        self.actor = actor
        self.actor_processor = actor_processor
        for param in self.pigai_model.parameters():
            param.requires_grad = False

        # self.pigai_batch = 1
        
    def generate_revision(self, query_list, original_answer_list, imgpath_list, critique_list, model, processor, num_gen_revision):
        """Generate critique for the erroneous steps"""
        messages_list = []
        text_list = []
        inputs=[]
        for query, original_answer, imgpath, critique in zip(query_list, original_answer_list, imgpath_list, critique_list):
            critique = critique.replace('<|im_end|>','')
            try:
                critique = eval(critique)
                cri_score = critique['score']
                critique_l = critique['critique']
                cri_str = ''
                for cri_id in range(len(critique_l)):
                    cri_str+=f"""{cri_id+1}. {critique_l[cri_id]}\n"""
            except:
                cri_str = str(critique)#.split('''"score"''')[0].split('{')[0]
            user_critic='''Please refine the model-generated response by considering the evaluation comments provided below. 
    These comments serve as guidance to enhance the accuracy, relevance, and comprehensiveness of the response. 
    While incorporating the suggestions, ensure that the final output remains natural and contextually appropriate, rather than strictly adhering to the comments.

    User Input: 
    <User Input Start>
    <|vision_start|><|image_pad|><|vision_end|>
    <User Input>
    <User Input End>

    Model Generated Content: 
    <Model Generated Content Start> 
    <Model Generated Content> 
    <Model Generated Content End>

    Evaluation Comments: 
    <Evaluation Comments Start> 
    <Evaluation Comments> 
    <Evaluation Comments End>

    Based on the evaluation comments, refine the model's response while maintaining natural flow and coherence. 
    Use the comments as helpful suggestions rather than strict rules. The final response should be improved but still retain its original style and intent. Output the revised content directly (Note: the output should not include any prefixes or suffixes like <-Start> or <-End>).'''.replace('<User Input>', query).replace('<Model Generated Content>', original_answer.replace('<|im_end|>', '')).replace('<Evaluation Comments>',cri_str)
            messages = [
            {"role": "user", "content": [ {"type": "image", "image": imgpath}, {"type": "text", "text": user_critic}]},

        ]

            # Generate response
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )+f"Let's think step by step."
            text = text.replace('user\n<|vision_start|><|image_pad|><|vision_end|>','user\n')
            images, videos = process_vision_info(messages)
            inputs += [{"prompt":text,"multi_modal_data": {"image": images[0]}}]
        # samplingparams = SamplingParams(max_tokens=2048, temperature=0, skip_special_tokens=False)
        inputs = expand_list(inputs, num_gen_revision)
        samplingparams = SamplingParams(max_tokens=2048, top_p=0.9, top_k=50, temperature=0.7, skip_special_tokens=False, repetition_penalty=1.05)
        generated_ids_trimmed = []
        outputs = model.generate(inputs, samplingparams, use_tqdm=False)
        for out in outputs:
            generated_ids_trimmed.append(out.outputs[0].token_ids)#cumulative_logprob
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )   
        return output_text



    def generate_revision_outputs(self, query_list, original_answer_list, imgpath_list, critique_list, model, processor, num_gen_revision, revision0):
        """Generate critique for the erroneous steps"""
        messages_list = []
        text_list = []
        inputs=[]
        for query, original_answer, imgpath, critique in zip(query_list, original_answer_list, imgpath_list, critique_list):
            critique = critique.replace('<|im_end|>','')
            try:
                critique = eval(critique)
                cri_score = critique['score']
                critique_l = critique['critique']
                cri_str = ''
                for cri_id in range(len(critique_l)):
                    cri_str+=f"""{cri_id+1}. {critique_l[cri_id]}\n"""
            except:
                cri_str = str(critique)#.split('''"score"''')[0].split('{')[0]
            user_critic='''Please refine the model-generated response by considering the evaluation comments provided below. 
These comments serve as guidance to enhance the accuracy, relevance, and comprehensiveness of the response. 
While incorporating the suggestions, ensure that the final output remains natural and contextually appropriate, rather than strictly adhering to the comments.

User Input:
<|vision_start|><|image_pad|><|vision_end|>
<User Input>

Model Generated Content:
<Model Generated Content>

Corrective Feedback:
<Corrective Feedback>


Based on the evaluation comments, refine the model's response while maintaining natural flow and coherence. 
Use the comments as helpful suggestions rather than strict rules. The final response should be improved but still retain its original style and intent. Output the revised content directly (Note: the output should not include any prefixes or suffixes like <-Start> or <-End>).'''.replace('<User Input>', query).replace('<Model Generated Content>', original_answer.replace('<|im_end|>', '')).replace('<Corrective Feedback>',cri_str)
            messages = [
            {"role": "user", "content": [ {"type": "image", "image": imgpath}, {"type": "text", "text": user_critic}]},

        ]

            # Generate response
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )+f"Let's think step by step."
            text = text.replace('user\n<|vision_start|><|image_pad|><|vision_end|>','user\n')
            images, videos = process_vision_info(messages)
            inputs += [{"prompt":text,"multi_modal_data": {"image": images[0]}}] * (num_gen_revision-1)


        samplingparams = SamplingParams(max_tokens=2048, top_p=1.0, top_k=50, temperature=1.0, skip_special_tokens=False, repetition_penalty=1.05)
        generated_ids_trimmed = []
        outputs = model.generate(inputs, samplingparams, use_tqdm=False)
        for out in outputs:
            generated_ids_trimmed.append(out.outputs[0].token_ids)#cumulative_logprob
        output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )   
        output_text.append(revision0)
        completions_prompt = [text + complete for complete in output_text]
        # completions_prompt.append(text + revision0)
        inputs = processor(
            text=text,
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        processor.tokenizer.padding_side = 'right' # 设置<|end_of_text|>添加到右侧(默认为左侧)
        completion_inputs = processor(
            text=completions_prompt,
            images=images * num_gen_revision,
            videos=None, 
            padding=True,
            return_tensors="pt",
        )
        labels = completion_inputs['input_ids'].clone()
        prompt_len = inputs['input_ids'].shape[1]
        labels[:, :prompt_len] = -100 # 将prompt部分值赋值为0
        # <|endoftext|>: 151643
        labels[labels == 151643] = -100 # 将所有<|endoftext|>处的值赋值为0
        return output_text, completion_inputs, labels  
    def generate_pigai_prompt_tokens(self, question, answer, target):
        processor = self.pigai_processor
        messages = []
        messages.append([
            {'role':'system', 'content':'You are a helpful assistant.'},
            {'role':'user', 'content':f"You are playing the role of a teacher. Based on the standard answer, you determine whether the student's response is correct. The question is: {question}, the student's response is: {answer}, and the standard answer is: {target}"},
            {'role':'assistant', 'content':' My grading result is: [score].'},
        ])
        chat_texts = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, continue_final_message=False
        )
        for chat_idx in range(len(chat_texts)):
            chat_text = chat_texts[chat_idx]
            if chat_text.endswith('\n'):
                chat_texts[chat_idx] = chat_texts[chat_idx][:-1]
        tokens = processor(
                    text=chat_texts,
                    padding=True,
                    return_tensors="pt",
                )
        return tokens
    def generate_pigai_prompt_tokens_batch(self, question, answer_list, target, processor):
        messages = []
        for answer in answer_list:
            messages.append([
                {'role':'system', 'content':'You are a helpful assistant.'},
                {'role':'user', 'content':f"You are playing the role of a teacher. Based on the standard answer, you determine whether the student's response is correct. The question is: {question}, the student's response is: {answer.replace('<|im_end|>', '')}, and the standard answer is: {target}"},
                {'role':'assistant', 'content':' My grading result is: [score].'},
            ])
        chat_texts = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, continue_final_message=False
        )
        for chat_idx in range(len(chat_texts)):
            chat_text = chat_texts[chat_idx]
            if chat_text.endswith('\n'):
                chat_texts[chat_idx] = chat_texts[chat_idx][:-1]
        tokens = processor(
                    text=chat_texts,
                    padding=True,
                    return_tensors="pt",
                )
        return tokens

    def pigai_batch(self, question, answer_list, target, pigai_model, pigai_processor):
        value_list_all = []
        answer_list_split = split_list(answer_list, 8)
        for answer_list in answer_list_split:
            pigai_tokens = self.generate_pigai_prompt_tokens_batch(question, answer_list, target, pigai_processor)
            pigai_tokens = pigai_tokens.to(pigai_model.device)
            with torch.no_grad():
                logits = pigai_model(**pigai_tokens).logits
                last_token_index = (pigai_tokens['attention_mask'] != 0).sum(-1) - 1
                logits = logits[:, :, 0][list(range(len(logits))), last_token_index]
                value_list = torch.sigmoid(logits).tolist()
                value_list_all += value_list
        # value_list_all += value_list
        value_list_all_ = [1 if x > 0.5 else 0 for x in value_list_all]
        return np.mean(value_list_all_), value_list_all_
    def pigai(self, question, completion, target):
        """
        Args:
            question (str): query
            completions (str): Generated outputs
            solution (str): Ground Truths.

        Returns:
            float: Reward scores
        """
        pigai_model = self.pigai_model
        pigai_processor = self.pigai_processor
        pigai_tokenizer = self.pigai_tokenizer

        
        pigai_value = None
        answer = completion
        
        pigai_tokens = self.generate_pigai_prompt_tokens(question, answer, target)
        pigai_tokens = pigai_tokens.to(pigai_model.device)
        with torch.no_grad():
            logits = pigai_model(**pigai_tokens).logits
            last_token_index = (pigai_tokens['attention_mask'] != 0).sum(-1) - 1
            logits = logits[:, :, 0][list(range(len(logits))), last_token_index]
            pigai_value = torch.sigmoid(logits).item()

        return pigai_value

    def generate_actor_output(self, completions, inputs, labels, reward_list):
        logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
        # logits_to_keep = max([len(logp) for logp in old_per_token_logps])
        inputs['logits_to_keep'] = logits_to_keep # completion的最大长度
        inputs['completion_mask'] = labels[:, -logits_to_keep:] != -100

        old_per_token_logps = torch.zeros((1,1))
        ref_per_token_logps = torch.zeros((1,1)) # 如果后续需要添加KL散度的时候需要再算base model的KL散度

        # 计算reward
        rewards_per_func = torch.zeros((len(reward_list), 2))
        rewards_per_func[:, 0] = torch.tensor(reward_list, dtype=torch.float32)
        # pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        # rewards_per_func[:, 1] = torch.tensor([1.0 if re.match(pattern, complete.replace('<|im_end|>', ''), re.DOTALL | re.MULTILINE) else 0.0 for complete in completions], dtype=torch.float32)
        rewards_per_func[:, 1] = torch.tensor([1.0 if 'Final answer:' in complete else 0.0 for complete in completions], dtype=torch.float32)
        accuracy_reward = rewards_per_func[:, 0] # 第一个reward func为Accuracy Reward
        rewards_per_func = rewards_per_func
        # Apply weights to each reward function's output and sum
        # rewards = rewards_per_func.sum(dim=1)
        rewards = rewards_per_func[:,0] + rewards_per_func[:,1]*0.0
        self.num_generation = len(reward_list)
        # 计算advantages
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, len(completions)).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, len(completions)).std(dim=1)
        # 正则化, 来计算GRPO中的Advantages
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(len(completions), dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(len(completions), dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        

        # input_ids, attention_mask, pixel_values, image_grid_thw, completion_mask, old_per_token_logps, ref_per_token_logps, advantages
        # per_token_logps需要在模型更新的时候计算保存梯度, 所以这里不用保存
        # 将所有tensor放到cpu上, 再保存到文件
        inputs = inputs.to('cpu')
        # old_per_token_logps = [old_per_token_logp.to('cpu') for old_per_token_logp in old_per_token_logps]
        # old_per_token_logps = pad_sequence(old_per_token_logps, batch_first=True, padding_value=0)#.reshape(-1)
        assert inputs['input_ids'].shape[0] == inputs['completion_mask'].shape[0]
        assert inputs['logits_to_keep'] == inputs['completion_mask'].shape[1]
        ref_per_token_logps = ref_per_token_logps.to('cpu')
        advantages = advantages.to('cpu')
        outputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'image_grid_thw': inputs['image_grid_thw'],
            'logits_to_keep': inputs['logits_to_keep'],
            'completion_mask': inputs['completion_mask'],
            'old_per_token_logps': old_per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'advantages': advantages,
            'rewards_per_func': rewards_per_func
        }
        return outputs, accuracy_reward

    def __call__(self, completions, ori_reward, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        original_answer = kwargs['original_answer']
        question = kwargs['question']
        imgpath = kwargs['imgpath']
        target = kwargs['target']
        query = kwargs['query']
        num_gen_revision = 1
        rewards = []
        correction_identity = []
        actor_outputs = []
        reward_dict = dict()
        rewards = [0] * len(completions)
        rest_ids = []
        rest_completions = []
        for content_id, content in enumerate(completions):
            if ori_reward == 1:
                if 'no corrections needed' in content.lower():
                    rewards[content_id]=1.0
                    correction_identity.append(1.0)
                    continue
                else:
                    rewards[content_id]=0.0
                    correction_identity.append(0.0)
                    continue
            else:
                if 'no corrections needed' in content.lower():
                    rewards[content_id]=0.0
                    correction_identity.append(0.0)
                    continue
            rest_ids.append(content_id)
            correction_identity.append(1.0)
            rest_completions.append(content.replace('<|im_end|>',''))
        if len(rest_ids)==0:
            return rewards, correction_identity, [], []
        revision_list= self.generate_revision([query]*len(rest_completions), [original_answer]*len(rest_completions), [imgpath]*len(rest_completions), rest_completions, self.actor, self.actor_processor, num_gen_revision)
        reward, reward_list = self.pigai_batch(question, revision_list, target, self.pigai_model, self.pigai_processor)

        actor_completions = []
        actor_revision_list = []
        for rew, rest_id, rest_completion, revision in zip(reward_list, rest_ids, rest_completions, revision_list):
            rewards[rest_id] = rew
            if rew == 1:
                actor_completions.append(rest_completion)
                actor_revision_list.append(revision)
        actor_outputs_revision = []
        accuracy_reward_list = []
        topk = 1
        if len(actor_completions) > 0:

            for cri_idx in range(min(topk, len(actor_completions))):

                revision_list, inputs, labels = self.generate_revision_outputs([query], [original_answer], [imgpath], [actor_completions[cri_idx]], self.actor, self.actor_processor, 32, actor_revision_list[cri_idx])
                

                reward, reward_list = self.pigai_batch(question, revision_list, target, self.pigai_model, self.pigai_processor)

                # reward_list.append(1)
                actor_output, accuracy_reward = self.generate_actor_output(revision_list, inputs, labels, reward_list)
                accuracy_reward_list.append(accuracy_reward)
                if not (torch.all(accuracy_reward == 0) or torch.all(accuracy_reward == 1)):

                    actor_outputs_revision.append(actor_output)


        return rewards, correction_identity, actor_outputs_revision, accuracy_reward_list
            

        


import json
class FormatORM():

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        # completions = [item.replace('<|im_end|>', '') for item in completions] # 将最后的<|im_end|>删除, 因为需要严格格式, 后面不能跟任何其他的内容
        # pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        # matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        # return [1.0 if match else 0.0 for match in matches]

        rewards = []
        for completion in completions:
            try:
                completion = completion.replace('<|im_end|>', '')
                # 先尝试解析为 JSON
                output = eval(completion)

                # 检查必须包含 critique 和 score 两个字段
                if not isinstance(output, dict):
                    rewards.append(0.0)
                    continue

                if "critique" not in output or "score" not in output:
                    rewards.append(0.0)
                    continue

                # 检查 critique 是 list，每一项是 str
                critique = output["critique"]
                if not isinstance(critique, list) or not all(isinstance(item, str) for item in critique):
                    rewards.append(0.0)
                    continue

                # 检查 score 是 1-5 的 int
                score = output["score"]
                if not isinstance(score, int) or not (score >= 1 and score <=5):
                    rewards.append(0.0)
                    continue

                # 如果全部符合，奖励 1.0
                rewards.append(1.0)
            except Exception:
                # 解析失败，直接 0
                rewards.append(0.0)
        return rewards


def generate_pigai_prompt_tokens(question, answer_list, target, processor):
    messages = []
    for answer in answer_list:
        messages.append([
            {'role':'system', 'content':'You are a helpful assistant.'},
            {'role':'user', 'content':f"You are playing the role of a teacher. Based on the standard answer, you determine whether the student's response is correct. The question is: {question}, the student's response is: {answer.replace('<|im_end|>', '')}, and the standard answer is: {target}"},
            {'role':'assistant', 'content':' My grading result is: [score].'},
        ])
    chat_texts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, continue_final_message=False
    )
    for chat_idx in range(len(chat_texts)):
        chat_text = chat_texts[chat_idx]
        if chat_text.endswith('\n'):
            chat_texts[chat_idx] = chat_texts[chat_idx][:-1]
    tokens = processor(
                text=chat_texts,
                padding=True,
                return_tensors="pt",
            )
    return tokens









