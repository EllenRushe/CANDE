#!/bin/bash -l

#SBATCH -N 4
#SBATCH -n 4 
#SBATCH --output=supervised.log

cd $SLURM_SUBMIT_DIR
source  activate contextNet
srun -N 1 -n 1 python main_supervised.py MLP_32 &
srun -N 1 -n 1 python main_supervised.py MLP_64 &
srun -N 1 -n 1 python main_supervised.py MLP_128 &
srun -N 1 -n 1 python main_supervised.py MLP_256 &
wait
