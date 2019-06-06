#!/bin/bash
#SBATCH -n 1
#SBATCH --array=20,73,82,85
#SBATCH --job-name=activations
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --mem=8GB
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm
#SBATCH -c 2


hostname

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

cd /om/user/sanjanas/eccentricity-crop
echo "COMMAND LINE VERSION"
python -V
singularity exec -B /om2:/om2 -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/sanjanas/eccentricity-crop/main.py $((${SLURM_ARRAY_TASK_ID} + 8650)) activations
