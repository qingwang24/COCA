source ./.bashrc
export script_dir=$(pwd)
export dir_name=$(basename "$PWD")
# cd ../../

source /dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/bin/activate swift_ycp

# 数据采样
# model_path=/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-2B-Instruct/ # 初始模型
# model_path=/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-7B-Instruct/

export CUDA_HOME='/opt/lib/cuda-12.6/'
export PATH=/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/envs/swift_ycp/bin:$PATH
# export PATH=/opt/dls_cli:/dmx-csy-home/intern/qkchang/.vscode-server/bin/8b3775030ed1a69b13e4f4c628c612102e30a681/bin/remote-cli:/opt/dls_cli:/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/envs/swift_ycp/bin:/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:
export PYTHONPATH=/dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/envs/swift_ycp/lib/python3.10/site-packages
export VLLM_ATTENTION_BACKEND="XFORMERS"
actor_model_path=/dmx-csy-mix01/cog3/permanent/qkchang/pretrained_models/Qwen2.5-VL-7B-Instruct/
critic_model_path=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/models/pretrained_critic_2.5vl_base

# actor_model_path=/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-2B-Instruct/
# critic_model_path=/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2-VL-2B-Instruct/

pigai_model_path=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/models/pigai_model/checkpoint-467/
# output_dir=/train21/cog8/permanent/qkchang/R1_Zero/experiments/${dir_name}
output_dir=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/experiments/GRPO/qwen2.5/actor_only/
# output_dir=/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/rlhf/experiments/debug
test_dataset=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/dataset/train_filter_10w.json
bash_path=/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/code/GRPO/swift_0707_qwen2.5vl_actor/scripts/iter_v1
num_generation=16
gpu_memory_utilization=0.4
per_device_train_batch_size=1
iter_num=8
deepspeed=zero1
begin_id=0 # 从0开始
gradient_accumulation_steps=8
check_interval_seconds=20
duration_seconds=180
skip_first_sample=False # 不起作用了, 自动检测是否需要跳过采样阶段
# 保证: per_device_train_batch_size * device = 2 * 8 = 16 / 16 = 1
# batch_size: 304 * 7 * 2 = 4256, 
# step = 20000/2 (query num) * 16 (G) / batch_size = 160000 / 4256 = 50 



# if [ "${begin_id}" -eq 0 ]; then
#     model_path=${model_path}
# else
#     model_path=${output_dir}/model_output/iter_$((begin_id-1))/last/
# fi



export NGPUS=8
export NNODES=5
export MASTER_PORT=10025
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7



python3 /dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/code/GRPO/swift_0707_qwen2.5vl_actor/off_policy_GRPO/grpo_iter_new.py\
        --output_dir ${output_dir} \
        --actor_model_path ${actor_model_path} \
        --critic_model_path ${critic_model_path} \
        --pigai_model_path ${pigai_model_path} \
        --test_dataset ${test_dataset} \
        --bash_path ${bash_path} \
        --num_generation ${num_generation} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --gpu_memory_utilization ${gpu_memory_utilization} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --iter_num ${iter_num} \
        --begin_id ${begin_id} \
        --deepspeed ${deepspeed} \
        --check_interval_seconds ${check_interval_seconds} \
        --duration_seconds ${duration_seconds} \
        --skip_first_sample ${skip_first_sample} \
        --NGPUS ${NGPUS} \
        --NNODES ${NNODES} \
        --MASTER_PORT ${MASTER_PORT}





