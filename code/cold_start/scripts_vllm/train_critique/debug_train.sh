source ./.bashrc

nvidia-smi
export script_dir=$(pwd)
export dir_name=$(basename "$PWD")
source /dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/bin/activate swift_ycp

set -x 
export CUDA_HOME='/opt/lib/cuda-12.6/'
export PATH=/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/envs/swift_ycp/bin:$PATH
# export PATH=/opt/dls_cli:/dmx-csy-home/intern/qkchang/.vscode-server/bin/8b3775030ed1a69b13e4f4c628c612102e30a681/bin/remote-cli:/opt/dls_cli:/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/envs/swift_ycp/bin:/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:
export PYTHONPATH=/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/envs/swift_ycp/lib/python3.10/site-packages
export VLLM_ATTENTION_BACKEND="XFORMERS"
# export MODELSCOPE_CACHE=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/Cache/rlhf/dataset/tmp_${dir_name}
# export HF_DATASETS_CACHE=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/Cache/rlhf/dataset/tmp_${dir_name}
export PYTHONPATH=$(pwd) # 确定pythonpath
# export TRITON_CACHE_DIR=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/Cache/rlhf/dataset/triton_cache/${RANK}
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_NTHREADS=4
export NCCL_TIMEOUT=600 # 默认30s
export TORCHELASTIC_AGENT_EXIT_BARRIER_TIMEOUT=60000 # 增大等待时间


export CXX=g++


export TRITON_CACHE_DIR=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/code/sft/.triton/autotune
# export TORCH_EXTENSIONS_DIR=/home5/intern/ycpan4/.cache/torch_extensions/py39_cu121/

cd ../../runner/vllm/


# export NCCL_DEBUG=info
# export NCCL_SOCKET_IFNAME=eno2.100
# export TRITON_CACHE_DIR=/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/.triton/autotune/
export NGPUS=8
export NNODES=4
# export CUDA_VISIBLE_DEVICES=0,1
save_steps=500
export model_name_or_path=../../libs/configs
export output_dir=../../experiments/train_critique/qwen2.5vl_3B
export dataloader_num_workers=0
export gradient_accumulation_steps=64
export num_train_epochs=1
export learning_rate=1e-5
export warmup_ratio=0.05
export master_port=10025
export per_device_train_batch_size=1

        # --deepspeed ../ds_config/zero0.json \
if [[ $NNODES -gt 1 ]]; then
        torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$master_port \
        train.py \
        --deepspeed ../../ds_config/zero0.json \
        --model_name_or_path $model_name_or_path \
        --output_dir $output_dir \
        --do_train \
        --save_strategy steps \
        --save_steps $save_steps \
        --logging_strategy steps \
        --logging_steps 5 \
        --per_device_train_batch_size $per_device_train_batch_size \
        --learning_rate $learning_rate \
        --lr_scheduler_type constant \
        --num_train_epochs $num_train_epochs \
        --dataloader_num_workers $dataloader_num_workers \
        --warmup_ratio $warmup_ratio \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --bf16 \
        --ignore_data_skip True \
        --report_to none
else
        torchrun --nproc_per_node=$NGPUS --master_port=$master_port \
        train.py \
        --deepspeed ../../ds_config/zero0.json \
        --model_name_or_path $model_name_or_path \
        --output_dir $output_dir \
        --do_train \
        --save_strategy steps \
        --save_steps $save_steps \
        --logging_strategy steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --logging_steps 5 \
        --learning_rate $learning_rate \
        --lr_scheduler_type constant \
        --num_train_epochs $num_train_epochs \
        --dataloader_num_workers $dataloader_num_workers \
        --warmup_ratio $warmup_ratio \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --bf16 \
        --ignore_data_skip True \
        --report_to none
fi
