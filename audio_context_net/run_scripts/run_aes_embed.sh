#!/bin/bash -l

#SBATCH -N 4
#SBATCH -n 4 
#SBATCH --output=aes_embed.log

cd $SLURM_SUBMIT_DIR
source  activate contextNet

srun -N 1 -n 1 ./run_scripts/ae_sound_film_embed_32.sh  3 &
srun -N 1 -n 1 ./run_scripts/ae_sound_film_embed_64.sh  3 &
srun -N 1 -n 1 ./run_scripts/ae_sound_film_embed_128.sh  3  &
srun -N 1 -n 1 ./run_scripts/ae_sound_film_embed_256.sh  3  &

wait

