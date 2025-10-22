#!/bin/bash

#SBATCH --job-name=ddm_fitting
#SBATCH --array=0-44
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=480


module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/vaibhav/nhp-prior-data:/mnt/data,/mmfs1/home/vthakur/decision-models/:/src/ \
    /gscratch/walkerlab/vaibhav/decision-models.sif /usr/bin/python3 \
    /src/scripts/2.0-block-data-fitting-slurm.py --session_id $SLURM_ARRAY_TASK_ID
