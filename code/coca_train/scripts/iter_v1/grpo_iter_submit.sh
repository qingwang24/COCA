#! /bin/bash
dir_name=$(basename "$PWD")


bash_path=grpo_iter.sh
touch master_$(date +%Y%m%d_%H%M).log
ky exp submit \
     -a qkchang \
     -n train-master-v1-shl \
     -d "${dir_name}" \
     --dataType local \
     --dataPath /train34/mmu/permanent/cxqin/zrzhang6/ChartQA/qkchang/code/prm/v1 \
     --modelVersion 175B-v0428 \
     --modelName vdu1.1.0.6.1-$(date +%Y%m%d_%H%M%S) \
     --modelPath /train21/mmu/permanent/zrzhang6/pfhu6/code/sft-1/lmexpand/experiments \
     -i reg.deeplearning.cn/ky/shell-ubuntu-dlp:22.04 \
     -e ${bash_path} \
     -l master_$(date +%Y%m%d_%H%M).log \
     --useGpu \
     --remoteDebug \
     -g 1 \
     -m 200 \
     -t PtJob \
     -w 1 \
     --proID 253 \
     -k HopperH200-NVLINK-141GB \
     -r dlp3-cog3-cogllm-2-h200-reserved 
  


