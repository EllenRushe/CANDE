#!/bin/bash -l

#SBATCH -N 3
#SBATCH -n 3 
#SBATCH --output=aes.log

cd $SLURM_SUBMIT_DIR
source  activate contextNet

srun -N 1 -n 1 ./run_scripts/ae_all.sh 30 &
srun -N 1 -n 1 ./run_scripts/ae_film_one_hot.sh 30 &
srun -N 1 -n 1 ./run_scripts/aes.sh 30  &
wait

