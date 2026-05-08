source /train1/cog8/permanent/zrzhang6/anaconda3/bin/activate swift
set -x 
# model_path=/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/DeepSeek-R1-Distill-Qwen-1.5B/
model_path=/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-1.5B-Instruct/

# 22GB
CUDA_VISIBLE_DEVICES=4 \
swift sft \
    --model ${model_path} \
    --train_type full \
    --dataset './dataset/self-recognition/' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot



