#! /bin/bash
dir_name=$(basename "$PWD")

bash_path=self_play_2_5.sh
touch $(date +%Y%m%d_%H%M).log
touch $(date +%Y%m%d_%H%M).err
ky exp submit \
     -a zrzhang6 \
     -n infer-selfplay \
     -d "${dir_name}" \
     --dataType local \
     --dataPath /train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/critique/dataset \
     --modelVersion 175B-v0428 \
     --modelName vdu1.1.0.6.1-$(date +%Y%m%d_%H%M%S) \
     --modelPath /train21/mmu/permanent/zrzhang6/pfhu6/code/sft-1/lmexpand/experiments \
     -i reg.deeplearning.cn/dlaas/cv_dist:0.1 \
     -e ${bash_path} \
     -l $(date +%Y%m%d_%H%M).log \
     -o $(date +%Y%m%d_%H%M).err \
     --useGpu \
     --remoteDebug \
     -g 8 \
     -m 680 \
     -t PtJob \
     -w 2 \
     --proID 253 \
     -k TeslaA800-NVLINK-80GB \
     -r dlp3-cog8-cogllm-4-reserved     


