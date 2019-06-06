#!/bin/bash
#SBATCH -n 1
#SBATCH --array=0,1,4,8,12,16,20,24,30,36,41,48,56,62,67,72,78,84,90,97,105,112,119,125,131,138,145,152,158,165,172
#SBATCH --job-name=activations
#SBATCH --mem=4GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -c 2
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm


hostname

# /om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

cd /om/user/sanjanas/eccentricity-crop
echo "COMMAND LINE VERSION"
python -V
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/sanjanas/eccentricity-crop/main.py $((${SLURM_ARRAY_TASK_ID} + 9210)) 
echo "sanji"
