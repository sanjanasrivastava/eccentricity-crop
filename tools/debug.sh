#!/bin/bash
#SBATCH -n 2
#SBATCH --array=0
#SBATCH --job-name=minimal
#SBATCH --mem=80GB
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

cd /om2/user/sanjanas/eccentricity-crop
singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python -V

