#!/bin/bash
#SBATCH -n 1
#SBATCH --array=0-35
#SBATCH --job-name=fcn_vanilla_random
#SBATCH --mem=4GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -c 2
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm


hostname

# /om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

cd /om/user/sanjanas/eccentricity-crop
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/sanjanas/eccentricity-crop/main.py $((${SLURM_ARRAY_TASK_ID} + 9933)) 
