
source /train1/cog8/permanent/zrzhang6/anaconda3/bin/activate swift
set -x 
model_path=/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/DeepSeek-R1-Distill-Qwen-1.5B/

# 使用交互式命令行进行推理
# CUDA_VISIBLE_DEVICES=3 \
# swift infer \
#     --model ${model_path} \
#     --stream true \
#     --temperature 0 \
#     --max_new_tokens 10000


# merge-lora并使用vLLM进行推理加速
CUDA_VISIBLE_DEVICES=1 \
swift infer \
    --model ${model_path} \
    --stream true \
    --infer_backend vllm \
    --max_model_len 2048 \
    --temperature 0 \
    --max_new_tokens 10000

