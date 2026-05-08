from vllm import LLM, SamplingParams
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import json
from qwen_vl_utils import process_vision_info
import torch
from trl.trainer.utils import selective_log_softmax
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
critic_path = '/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/rlhf/model/pretrained_critic'
actor_path = '/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-2B-Instruct'
actor = LLM(
        model=actor_path,
        device=f'cuda:5',
        gpu_memory_utilization=0.4,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 8, "video": 8},
        enforce_eager=False
    )
model = Qwen2VLForConditionalGeneration.from_pretrained(actor_path).eval().to('cuda:5')
model.eval()
test_dataset = json.load(open('/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/rlhf/dataset/ori/train_filter.json'))
processor = AutoProcessor.from_pretrained(actor_path)
for data_item in test_dataset:

    prompt = data_item['prompt']
    query = extract_prompt_content(prompt)
    question = query.split('Question:')[-1].strip()
    inputs = []
    messages = [{"role": "user", 
                "content": [
                    {"type": "image", "image": data_item['imgpath']}, 
                    {"type": "text", "text": query}]
                }]
                # messages = [{"role": "user", "content": [{"type": "image", "image": imgpath}]}]
    text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )+'''Let's think step by step.'''
    images, videos = process_vision_info(messages)
    inputs += [{"prompt":text,"multi_modal_data": {"image": images[0]}}]
    temperature = 0.7
    samplingparams = SamplingParams(max_tokens=2048, temperature=temperature, top_p=0.9,top_k=50, skip_special_tokens=False, logprobs=0)
    # samplingparams = SamplingParams(max_tokens=2048, top_p=0.9,top_k=50,temperature=0.7, skip_special_tokens=False)
    generated_ids_trimmed = []
    with torch.no_grad():
        outputs = actor.generate(inputs, samplingparams, use_tqdm=False)
        old_per_token_logps=[]
        for out in outputs:
            completion_token_ids = out.outputs[0].token_ids
            generated_ids_trimmed.append(completion_token_ids)#cumulative_logprob

            prompt_token_ids = out.prompt_token_ids

            logits_to_keep = len(completion_token_ids)
            num_placeholder = out.multi_modal_placeholders['image'][0]['length']
            input_ids = prompt_token_ids + list(completion_token_ids)
            text = processor.tokenizer.decode(input_ids).replace('<|image_pad|>'*num_placeholder, '<|image_pad|>')
            inputs1 = processor(text=text, images=images, videos=None, return_tensors="pt").to(model.device)
            inputs1['input_ids'] = torch.tensor([input_ids])
            inputs1 = inputs1.to(model.device)
            logits = model(**inputs1).logits
            logits = logits[:, -(logits_to_keep + 1):-1, :]
            logits = logits/temperature
            input_ids1 = inputs1['input_ids'][:, -logits_to_keep:]

            old_per_token_logp = torch.tensor([list(outputs[0].outputs[0].logprobs[idx].values())[0].logprob for idx in range(len(outputs[0].outputs[0].logprobs))])
            old_per_token_logp1 = selective_log_softmax(logits, input_ids1)

            print(torch.exp(old_per_token_logp - old_per_token_logp1.to('cpu')))
            1
        # old_per_token_logps.append(old_per_token_logp)
        
    # output_text = processor.batch_decode(
    #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #     )
