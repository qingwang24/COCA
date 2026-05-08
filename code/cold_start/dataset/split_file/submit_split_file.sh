#! /bin/bash
dir_name=$(basename "$PWD")
ky exp submit \
     -a zrzhang6 \
     -n data-split \
     -d "${dir_name}" \
     --dataType local \
     --dataPath /train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/mcts_alpha/exp/data_path/ \
     --modelVersion 175B-v0428 \
     --modelName vdu1.1.0.6.1-$(date +%Y%m%d_%H%M%S) \
     --modelPath /train34/mmu/permanent/cxqin/zrzhang6/GeoMathAnswer/pretrained_models/Qwen2.5-VL-7B-Instruct \
     -i reg.deeplearning.cn/ky/cv_dist_openmpi:0.7_shiqiu \
     -e split_file.sh \
     -l split_file.log \
     -o split_file.err \
     --useGpu \
     --remoteDebug \
     -g 8 \
     -m 1920 \
     -t PtJob \
     -w 1 \
     --proID 253 \
     -k TeslaA800-NVLINK-80GB \
     -r dlp3-cog8-cogllm-4-reserved
     