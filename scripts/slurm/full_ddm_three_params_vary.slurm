#!/bin/bash

#SBATCH --job-name=ddm_fitting
#SBATCH --array=0-44
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=15:00:00

# # Optional but helpful for debugging/logging each file for each array task:
# # SBATCH -o /gscratch/walkerlab/vaibhav/logs/ddm_fit_%A_%a.out   # STDOUT log
# # SBATCH -e /gscratch/walkerlab/vaibhav/logs/ddm_fit_%A_%a.err   # STDERR log

# module load singularity
# singularity exec --writable-tmpfs --nv \
#     --bind /gscratch/walkerlab/vaibhav/nhp-prior-data:/mnt/data,/mmfs1/home/vthakur/decision-models/:/src/ \
#     /gscratch/walkerlab/vaibhav/decision-models.sif /usr/bin/python3 \
#     /src/scripts/3.0-data-fitting-three-params-vary-slurm.py --session_id $SLURM_ARRAY_TASK_ID

# All jobs append to one shared log file
#SBATCH -o /gscratch/walkerlab/vaibhav/logs/ddm_fit_master.log

{
    echo "========== JOB START =========="
    echo "Date: $(date)"
    echo "Job ID: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
    echo "Node: $(hostname)"
    echo

    module load singularity
    singularity exec --writable-tmpfs --nv \
        --bind /gscratch/walkerlab/vaibhav/nhp-prior-data:/mnt/data,/mmfs1/home/vthakur/decision-models/:/src/ \
        /gscratch/walkerlab/vaibhav/decision-models.sif /usr/bin/python3 \
        /src/scripts/3.0-data-fitting-three-params-vary-slurm.py --session_id "$SLURM_ARRAY_TASK_ID"

    echo
    echo "========== JOB END =========="
    echo "Date: $(date)"
    echo "Job ID: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
    echo "---------------------------------------------"
    echo
} 2>&1 | tee -a /gscratch/walkerlab/vaibhav/logs/ddm_fit_master_three_params_vary.log
