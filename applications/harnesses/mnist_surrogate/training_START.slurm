#!/bin/bash

# NOTE: Number of CPUs per GPU must be an even number since there are
# 2 threads per core. If an odd number is requested the next higher
# even number gets used. Also, there are 120Gb of memory on the
# node. However, if 30Gb per-job are requested for 4 jobs there are
# not enough resources. You may request up to 29G per-job.

#SBATCH --job-name=mnist_study<studyIDX>
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=8
#SBATCH --gpu-bind=closest
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-bind=local
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=study<studyIDX>_epoch<epochIDX>.out
#SBATCH --error=study<studyIDX>_epoch<epochIDX>.err

# Load the right environment
source $YOKE_CONDA/bin/activate $YOKE_TORCH

# First-run invocation
python <train_script> @study<studyIDX>_START.input
