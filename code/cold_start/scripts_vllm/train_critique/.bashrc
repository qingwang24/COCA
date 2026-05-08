# >>> pip source initialize >>>
alias pip='function _pip(){
if [ $1 = "search" ]; then
/opt/scheduler/hpc-slurm-official/bin/pip_ayers_search "$2";
else pip "$@";
fi;
};_pip'

# export PATH=/opt/lib/cuda-11.7.1/bin/:${PATH}
# export LD_LIBRARY_PATH=/opt/lib/cuda-11.7.1/lib64:{LD_LIBRARY_PATH}

export MODULEPATH="/opt/tool/modulefiles/"
module unload gcc
# module load gcc/7.3.0-os7.2
module load gcc/7.3.0-os7.2
module load cuda/11.8-cudnn-v8.9.0
export CUDA_HOME=/opt/lib/cuda-11.8/

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/train1/cog8/permanent/zrzhang6/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/train1/cog8/permanent/zrzhang6/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/train1/cog8/permanent/zrzhang6/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/train1/cog8/permanent/zrzhang6/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

alias rm=/opt/ayers_trash/bin/ayers_trash
umask 0022
alias rm=/opt/ayers_trash_$(uname -p)/bin/ayers_trash
export PATH=/home5/bitbrain/wlhu5/.local/git/bin:$PATH
