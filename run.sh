#!/usr/bin/env bash
#PBS -l nodes=1:ppn=9:gpus=1:skylake
#PBS -l partition=gpu
#PBS -l walltime=06:00:00
#PBS -l pmem=2gb
#PBS -A default_project

# Exit on errors :)
set -e

cd ~/SEM-MacridVAE

echo "Exporting variables"
export PATH="${VSC_DATA}/cuda11.7/bin:${PATH}"
export LD_LIBRARY_PATH="${VSC_DATA}/cuda-11.7/lib64:${LD_LIBRARY_PATH}"

echo "Activating conda"
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(conda 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${VSC_DATA}/miniconda3/etc/profile.d/conda.sh" ]; then
        . "${VSC_DATA}/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="${VSC_DATA}/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

echo "Activating conda environment"
conda activate semmacridvae


echo "Checking for GPU in Torch"
echo 'import torch; torch.cuda.current_device()' | python

echo "Running main.py"
python main.py --data amazon-clothing --model DisenEVAE --epochs 50 --batch_size 100 --device cuda
