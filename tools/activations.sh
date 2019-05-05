#!/bin/bash
#SBATCH -n 1
#SBATCH --array=34-39,274-279
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
# singularity exec -B /om2:/om2 -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg python -c 'import tensorflow; import sys; print("STRING CALL VERSION:", sys.version); print("SUCCESSFULLY IMPORTED TENSORFLOW")'
singularity exec -B /om2:/om2 -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/sanjanas/eccentricity-crop/main.py $((${SLURM_ARRAY_TASK_ID} + 7200)) changed_condition_activations
echo "sanji"
