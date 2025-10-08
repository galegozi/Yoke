#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu_debug
#SBATCH --reservation=gpu_debug
#SBATCH --time=02:00:00
#SBATCH --output=imgen210.out
#SBATCH --error=imgen210.err
#SBATCH --job-name=imgen210
#SBATCH --mail-user=galgal@lanl.gov
#SBATCH --mail-type=ALL
#SBATCH --account=w25_artimis_g
#SBATCH --ntasks-per-node=64
#SBATCH --constraint=gpu40
#SBATCH -vvv


sinfo  -o "%P %.24G %N"
srun /usr/bin/echo $CUDA_AVAILABLE_DEVICES

module load python/3.11-anaconda-2023.07
source activate
conda activate torch_ch_gpu_241112

export MKL_SERVICE_FORCE_INTEL=TRUE

export date00=`date`

python lsc_loderunner_create_gif.py --runs-dir ../harnesses/chicoma_lsc_loderunner-ch-subsampling/runs --npz-dir /lustre/scratch5/exempt/artimis/data/lsc240420/ --run-id 400 --embed-dim 96 --mode single --run-list 210
python lsc_loderunner_create_gif.py --runs-dir ../harnesses/chicoma_lsc_loderunner-ch-subsampling/runs --npz-dir /lustre/scratch5/exempt/artimis/data/lsc240420/ --run-id 400 --embed-dim 96 --mode chained --run-list 210
python lsc_loderunner_create_gif.py --runs-dir ../harnesses/chicoma_lsc_loderunner-ch-subsampling/runs --npz-dir /lustre/scratch5/exempt/artimis/data/lsc240420/ --run-id 400 --embed-dim 96 --mode timestep --run-list 210

python lsc_loderunner_create_gif.py --runs-dir ../harnesses/chicoma_lsc_loderunner-ch-subsampling/runs --npz-dir /lustre/scratch5/exempt/artimis/data/lsc240420/ --run-id 4208 --embed-dim 96 --mode single --run-list 210
python lsc_loderunner_create_gif.py --runs-dir ../harnesses/chicoma_lsc_loderunner-ch-subsampling/runs --npz-dir /lustre/scratch5/exempt/artimis/data/lsc240420/ --run-id 4208 --embed-dim 96 --mode chained --run-list 210
python lsc_loderunner_create_gif.py --runs-dir ../harnesses/chicoma_lsc_loderunner-ch-subsampling/runs --npz-dir /lustre/scratch5/exempt/artimis/data/lsc240420/ --run-id 4208 --embed-dim 96 --mode timestep --run-list 210

export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
