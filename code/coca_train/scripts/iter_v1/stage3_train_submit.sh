#! /bin/bash
dir_name=$(basename "$PWD")

NGPUS=$1
NNODES=$2
task_name=$3

bash_path=stage3_train.sh
touch train3_$(date +%Y%m%d_%H%M).log
ky exp submit \
     -a qkchang \
     -n train-grpo-shl-${task_name} \
     -d "${dir_name}" \
     --dataType local \
     --dataPath /train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/code/prm/v1 \
     --modelVersion 7B \
     --modelName vdu1.1.0.6.2-$(date +%Y%m%d_%H%M%S) \
     --modelPath /dmx-csy-mix01/cog3/permanent/qkchang/shliu19/experiments \
     -i reg.deeplearning.cn/ky/shell-ubuntu-dlp:22.04 \
     -e ${bash_path} \
     -l train3_$(date +%Y%m%d_%H%M).log \
     --useGpu \
     --remoteDebug \
     -g ${NGPUS} \
     -m 1000 \
     -t PtJob \
     -w ${NNODES} \
     --proID 1969 \
     -k HopperH200-NVLINK-141GB \
     -r dlp3-cog3-cogllm-2-h200-reserved    


