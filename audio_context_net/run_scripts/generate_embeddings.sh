#!/bin/bash -l

#SBATCH -N 4
#SBATCH -n 4 
#SBATCH --output=generate_embeddings.log
cd $SLURM_SUBMIT_DIR
source  activate contextNet
best_ckpt_32=$(cat 'logs/mlp_sound_32/MLP_sound_32.json' | python -c 'import json, sys; log=json.load(sys.stdin); print(log["best_val_epoch"])')
best_ckpt_64=$(cat 'logs/mlp_sound_64/MLP_sound_64.json' | python -c 'import json, sys; log=json.load(sys.stdin); print(log["best_val_epoch"])')
best_ckpt_128=$(cat 'logs/mlp_sound_128/MLP_sound_128.json' | python -c 'import json, sys; log=json.load(sys.stdin); print(log["best_val_epoch"])')
best_ckpt_256=$(cat 'logs/mlp_sound_256/MLP_sound_256.json' | python -c 'import json, sys; log=json.load(sys.stdin); print(log["best_val_epoch"])')

srun -N 1 -n 1 python generate_embedding.py --model_name MLP_sound_32 --ckpt_name "checkpoint_MLP_sound_32_epoch_$best_ckpt_32" &
srun -N 1 -n 1 python generate_embedding.py --model_name MLP_sound_64 --ckpt_name "checkpoint_MLP_sound_64_epoch_$best_ckpt_64" &
srun -N 1 -n 1 python generate_embedding.py --model_name MLP_sound_128 --ckpt_name "checkpoint_MLP_sound_128_epoch_$best_ckpt_128" &
srun -N 1 -n 1 python generate_embedding.py --model_name MLP_sound_256 --ckpt_name "checkpoint_MLP_sound_256_epoch_$best_ckpt_256" &

wait

