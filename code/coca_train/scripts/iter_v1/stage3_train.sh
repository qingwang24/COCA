source ./.bashrc
export script_dir=$(pwd)
export dir_name=$(basename "$PWD")
cd ../../

source /dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/bin/activate swift_ycp
nvidia-smi
set -x 
export CUDA_HOME='/opt/lib/cuda-12.6/'
export PATH=/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/envs/swift_ycp/bin:$PATH
# export PATH=/opt/dls_cli:/dmx-csy-home/intern/qkchang/.vscode-server/bin/8b3775030ed1a69b13e4f4c628c612102e30a681/bin/remote-cli:/opt/dls_cli:/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/envs/swift_ycp/bin:/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:
export PYTHONPATH=/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/envs/swift_ycp/lib/python3.10/site-packages
export VLLM_ATTENTION_BACKEND="XFORMERS"
export MODELSCOPE_CACHE=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/Cache/rlhf/dataset/tmp_${dir_name}
export HF_DATASETS_CACHE=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/Cache/rlhf/dataset/tmp_${dir_name}
export PYTHONPATH=$(pwd) # 确定pythonpath
export TRITON_CACHE_DIR=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/Cache/rlhf/dataset/triton_cache/${RANK}
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_NTHREADS=4
export NCCL_TIMEOUT=600 # 默认30s
export TORCHELASTIC_AGENT_EXIT_BARRIER_TIMEOUT=60000 # 增大等待时间



torchrun --nproc_per_node 8 --nnodes=4 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=10035 ./swift/cli/rlhf.py --rlhf_type grpo --model /dmx-csy-mix01/cog3/permanent/qkchang/pretrained_models/Qwen2.5-VL-7B-Instruct/ --external_plugins ./examples/train/grpo/plugin/plugin.py --reward_funcs format --use_vllm false --vllm_gpu_memory_utilization 0.4 --temperature 0.7 --top_p 1.0 --top_k 50 --train_type full --torch_dtype bfloat16 --dataset /dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/experiments/GRPO/qwen2.5/sample_ablation/top3/sample_output/sample_iter_0/dataset/actor --split_dataset_ratio 0 --max_completion_length 4096 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-06 --warmup_ratio 0 --lr_scheduler_type constant --gradient_accumulation_steps 64 --eval_strategy no --save_strategy epoch --save_total_limit 10 --logging_steps 3 --max_length 4096 --output_dir /dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/experiments/GRPO/qwen2.5/sample_ablation/top3/model_output/iter_0/actor --dataloader_num_workers 4 --dataset_num_proc 4 --num_generations 16 --repetition_penalty 1.05 --system ./prompt.txt --deepspeed zero1 --log_completions true --num_iterations 1 --async_generate false --beta 0 --add_version False --model_type qwen2_5_vl --create_checkpoint_symlink True --freeze_vit True