#!/usr/bin/bash -l
set -e


echo "Exporting variables"
export PATH="${VSC_DATA}/cuda11.7/bin:${PATH}"
export LD_LIBRARY_PATH="${VSC_DATA}/cuda-11.7/lib64:${LD_LIBRARY_PATH}"

echo "Activating conda"
conda activate semmacridvae

echo "Checking for GPU in Torch"
echo 'import torch; torch.cuda.current_device()' | python

echo "Running main.py with extra arguments $@"
python main.py --data amazon-clothing --model SEMMacridVAE --epochs 50 --batch_size 100 --device cuda $@

