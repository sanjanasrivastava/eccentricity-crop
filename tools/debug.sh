#!/bin/bash
#SBATCH -n 1
#SBATCH --array=0
#SBATCH --job-name=debug
#SBATCH --mem=4GB
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm


hostname

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

cd /om/user/sanjanas/eccentricity-crop
singularity exec -B /om2:/om2 -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg python -c 'import tensorflow'
echo ':)'
