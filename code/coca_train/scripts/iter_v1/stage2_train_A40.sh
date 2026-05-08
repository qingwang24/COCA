source ./.bashrc
export script_dir=$(pwd)
export dir_name=$(basename "$PWD")
cd ../../

source /train1/cog8/permanent/zrzhang6/anaconda3/bin/activate swift_new
nvidia-smi
set -x 

export MODELSCOPE_CACHE=/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/rlhf/dataset/tmp_${dir_name}
export HF_DATASETS_CACHE=/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/rlhf/dataset/tmp_${dir_name}
export PYTHONPATH=$(pwd) # 确定pythonpath
export TRITON_CACHE_DIR=/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/rlhf/experiments/triton_cache/${RANK}
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_NTHREADS=4
export NCCL_TIMEOUT=600 # 默认30s
export TORCHELASTIC_AGENT_EXIT_BARRIER_TIMEOUT=60000 # 增大等待时间



torchrun --nproc_per_node 4 --master_port=10025 ./swift/cli/rlhf.py --rlhf_type grpo --model /train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-2B-Instruct/ --external_plugins ./examples/train/grpo/plugin/plugin.py --reward_funcs format --use_vllm false --use_lmdeploy false --vllm_device auto --vllm_gpu_memory_utilization 0.4 --temperature 0.7 --top_p 1.0 --top_k 50 --num_infer_workers 1 --train_type full --torch_dtype bfloat16 --dataset /train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/rlhf/experiments/debug/sample_output/sample_iter_0/dataset/critic --split_dataset_ratio 0 --max_completion_length 4096 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-06 --warmup_ratio 0 --lr_scheduler_type constant --gradient_accumulation_steps 4 --eval_strategy no --save_strategy epoch --save_total_limit 10 --logging_steps 3 --max_length 4096 --output_dir /train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/rlhf/experiments/debug/model_output/iter_0/critic --dataloader_num_workers 4 --dataset_num_proc 4 --num_generations 8 --repetition_penalty 1.1 --system ./prompt.txt --deepspeed zero2 --log_completions true --num_iterations 1 --async_generate false --beta 0 --add_version False --create_checkpoint_symlink True