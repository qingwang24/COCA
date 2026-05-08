#! /bin/bash
dir_name=$(basename "$PWD")
ky exp submit \
     -a qkchang \
     -n train-cri-shl \
     -d train-cri-shl \
     --dataType local \
     --dataPath /train34/mmu/permanent/cxqin/zrzhang6/ChartQA/code/MCTS/mcts_alpha/exp/data_path/ \
     --modelVersion 3B \
     --modelName vdu1.1.0.6.2-$(date +%Y%m%d_%H%M%S) \
     --modelPath /dmx-csy-mix01/cog3/permanent/qkchang/shliu19/experiments \
     -i reg.deeplearning.cn/ky/shell-ubuntu-dlp:22.04 \
     -e debug_train.sh \
     -l train3B.log \
     -o train3B.err \
     --useGpu \
     --remoteDebug \
     -g 8 \
     -m 1920 \
     -t PtJob \
     -w 4 \
     --proID 1969 \
     -k HopperH200-NVLINK-141GB \
     -r dlp3-cog3-cogllm-2-h200-reserved  
     # -k TeslaA100-PCIE-48GB \
     # -r dlp3-superbrain-cogllm-reserved \
     # -s "dlp2-14-219 dlp2-14-220 dlp2-14-221 dlp2-14-223"
     
     