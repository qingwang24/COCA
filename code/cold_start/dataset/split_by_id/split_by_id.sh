source ./.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/train1/cog8/permanent/zrzhang6/anaconda3/envs/mcts_new/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home4/intern/zrzhang6/anaconda3/lib

nvidia-smi


cd /train34/cog8/permanent/bhwei2/pfhu6/shliu19/code/MCTS/critique/dataset/v4/code/split_by_id
source /train1/cog8/permanent/zrzhang6/anaconda3/bin/activate mcts_new

python split_by_id.py