#!/bin/bash
#SBATCH -n 2
#SBATCH --array=6
#SBATCH --job-name=minimal
#SBATCH --mem=80GB
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

cd /om/user/sanjanas/eccentricity-crop
singularity exec -B /om:/om --nv /cbcl/cbcl01/xboix/singularity/localtensorflow.img \
python /om/user/sanjanas/eccentricity-crop/main.py ${SLURM_ARRAY_TASK_ID}


