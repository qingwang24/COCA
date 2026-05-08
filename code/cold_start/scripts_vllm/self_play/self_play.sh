source /train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/code/prm/v1/scripts/inference/.bashrc
if [ -f /.dockerenv ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home4/intern/zrzhang6/anaconda3/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home4/intern/zrzhang6/anaconda3/envs/mcts_chartqa/lib/
fi

nvidia-smi
export script_dir=$(pwd)
export dir_name=$(basename "$PWD")

cd ../../runner/vllm
export PATH=/opt/lib/gcc-11.4.0/lib64:/home4/intern/ycpan4/miniconda3/envs/mcts_cp39/bin:/opt/lib/gcc-11.4.0/bin:/opt/lib/cuda-12.1/bin:$PATH
export CXX=g++
export CUDA_HOME=/opt/lib/cuda-12.1
# export LD_LIBRARY_PATH=/usr/lib64/:/opt/lib/gcc-11.4.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/lib/gcc-11.4.0/lib64:/home4/intern/ycpan4/miniconda3/envs/mcts_cp39/bin:/opt/lib/gcc-11.4.0/bin:/opt/lib/cuda-12.1/bin:$LD_LIBRARY_PATH
export PYTHONPATH=/home5/intern/ycpan4/miniconda3/envs/mcts_cp39/lib/python3.9/site-packages/:$PYTHONPATH
export TRITON_CACHE_DIR=/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/.triton/autotune
source /home4/intern/ycpan4/miniconda3/bin/activate  mcts_cp39


output_dir=/train21/cog8/permanent/qkchang/shliu19/critique/code/v5/experiments/self_play/qwen2.5vl_v6/checkpoint-4388/mmstar_greedy
model_path=/train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-VL-7B-Instruct
pigai_model_path=/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/experiments/pigai/v1/train_v2/checkpoint-467/
ORM_model_path=/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/experiments/prm/v2/train_valid_v1/checkpoint-4407/
# critique_model_path=/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/v2/experiments/train_critique/v2_1/filtered_1/checkpoint-2192
# critique_model_path=/train21/cog8/permanent/qkchang/shliu19/critique/code/v5/experiments/train_critique/checkpoint-3500
critique_model_path=/train21/cog8/permanent/qkchang/shliu19/critique/code/v5/experiments/train_critique/qwen2.5vl_v6/checkpoint-4388
# critique_model_path=/train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/code/v3/experiments/train_critique/v2
# dataset_path=/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/code/prm/v1/dataset/direct/test_val_all_direct_process.json
dataset_path=/train21/cog8/permanent/qkchang/shliu19/dataset/Benchmark/test_final.json
# dataset_path="/train34/cog8/permanent/bhwei2/pfhu6/shliu19/datasets/mcts_dataset/ScienceQA/ScienceQA_mm.json"
# dataset_path="/train34/cog8/permanent/bhwei2/pfhu6/shliu19/datasets/mcts_dataset/MathVerse/MathVerse/MathVerse_mm_direct.json"
# dataset_path=/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/datasets/mcts_dataset/test_mcts_shuffle.json
# dataset_path=/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/dataset/mcts_for_critique/qwen2vl/train_fillin.json
# dataset_path=/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/dataset/mcts_train_2_5.json.json
# dataset_path=/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/dataset/mcts_for_critique/qwen2.5vl/train_fillin_current_withrollout.json
# MathVision
# dataset_path=/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/datasets/mcts_dataset/MathVision/MathVision_all.json
top_p=0.9
top_k=1
temperature=0.7
max_len=2048

best_of_n=1
qwen2vl_infer_batch=5

export NGPUS=8
export NNODES=4
export master_port=10025
# export CUDA_VISIBLE_DEVICES=4,5


if [[ $NNODES -gt 1 ]]; then
    torchrun --nproc_per_node $NGPUS --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        self_play.py \
        --output_dir ${output_dir} \
        --model_path ${model_path} \
        --pigai_model_path ${pigai_model_path} \
        --critique_model_path ${critique_model_path} \
        --dataset_path ${dataset_path} \
        --qwen2vl_infer_batch ${qwen2vl_infer_batch} \
        --best_of_n ${best_of_n} \
        --top_p ${top_p} \
        --top_k ${top_k} \
        --temperature ${temperature} \
        --max_len ${max_len}

else
	torchrun --nproc_per_node=$NGPUS --master_port=$master_port \
        self_play.py \
        --output_dir ${output_dir} \
        --model_path ${model_path} \
        --pigai_model_path ${pigai_model_path} \
        --critique_model_path ${critique_model_path} \
        --dataset_path ${dataset_path} \
        --qwen2vl_infer_batch ${qwen2vl_infer_batch} \
        --best_of_n ${best_of_n} \
        --top_p ${top_p} \
        --top_k ${top_k} \
        --temperature ${temperature} \
        --max_len ${max_len}
fi


