from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import torch

img_path = '/train34/cog8/permanent/bhwei2/pfhu6/qkchang/RL/swift/asset/wechat.png'
messages = [
                {"role": "user", "content":[{"type": "image", "image": img_path}, {"type": "text", "text": f"Answer"} ]}
            ]

multi_messages = [messages, messages]

processor = AutoProcessor.from_pretrained("/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-2B-Instruct")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
multi_text = processor.apply_chat_template(multi_messages, tokenize=False, add_generation_prompt=True)
multi_image_inputs, _ = process_vision_info(multi_messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=None,
    padding=True,
    return_tensors="pt",
)
pixel_values = inputs['pixel_values']
print(pixel_values.shape)

multi_inputs = processor(
    text=multi_text,
    images=multi_image_inputs,
    videos=None,
    padding=True,
    return_tensors="pt",
)
multi_pixel_values = multi_inputs['pixel_values']
print(multi_pixel_values.shape)
batch_size = 2
multi_batch_pixel_values = multi_pixel_values.view(batch_size, -1, pixel_values.shape[1])
print(torch.all(multi_batch_pixel_values[0] == pixel_values))

print()









