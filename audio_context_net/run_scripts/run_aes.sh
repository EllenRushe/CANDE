#!/bin/bash -l

#SBATCH -N 3
#SBATCH -n 3 
#SBATCH --output=aes.log

cd $SLURM_SUBMIT_DIR
source  activate contextNet

srun -N 1 -n 1 ./run_scripts/ae_sound_all.sh 3 &
srun -N 1 -n 1 ./run_scripts/ae_sound_film_one_hot.sh 3 &
srun -N 1 -n 1 ./run_scripts/aes_sound.sh 3  &
wait

