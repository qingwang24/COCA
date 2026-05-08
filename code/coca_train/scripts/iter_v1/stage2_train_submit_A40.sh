#! /bin/bash
dir_name=$(basename "$PWD")

NGPUS=$1
NNODES=$2
task_name=$3

bash_path=stage2_train_A40.sh
touch train_$(date +%Y%m%d_%H%M).log
ky exp submit \
     -a zrzhang6 \
     -n train-grpo-sh-${task_name} \
     -d "${dir_name}" \
     --dataType local \
     --dataPath /train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/code/prm/v1 \
     --modelVersion 175B-v0428 \
     --modelName vdu1.1.0.6.1-$(date +%Y%m%d_%H%M%S) \
     --modelPath /train21/mmu/permanent/zrzhang6/pfhu6/code/sft-1/lmexpand/experiments \
     -i reg.deeplearning.cn/ky/cv_dist_openmpi:0.7_shiqiu \
     -e ${bash_path} \
     -l train_$(date +%Y%m%d_%H%M).log \
     --useGpu \
     --remoteDebug \
     -g ${NGPUS} \
     -m 1000 \
     -t PtJob \
     -w ${NNODES} \
     --proID 253 \
     -k TeslaA100-PCIE-48GB \
     -r dlp3-superbrain-cogllm-reserved
     # -x "dlp2-12-092"     


