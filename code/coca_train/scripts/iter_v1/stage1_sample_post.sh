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



torchrun --nproc_per_node 8 --nnodes=4 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=10035 ./off_policy_GRPO/sample_post_process.py --output_dir /dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/experiments/GRPO/qwen2.5/sample_ablation/top3/sample_output --actor_model_path /dmx-csy-mix01/cog3/permanent/qkchang/pretrained_models/Qwen2.5-VL-7B-Instruct/ --critic_model_path /dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/models/pretrained_critic_2.5vl_base/ --pigai_model_path /dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/models/pigai_model/checkpoint-467/ --dataset_path /dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/dataset/train_filter_10w_sample_0.json --qwen2vl_infer_batch 1 --num_generation 16 --top_p 1.0 --top_k 50 --temperature 0.7 --max_len 4096 --gpu_memory_utilization 0.4 --exp_id sample_iter_0