#!/bin/bash
NUM_ITERS=$1
if [ $# -lt 1 ]
then
	echo "Need to pass number of iterations."
fi
# Runs from 1 to $NUM_ITERS inclusive
for i in $(seq "$NUM_ITERS")
do 
	python main.py AE_FiLM_embed_256 "" $i
	val_json="logs/ae_film_embed_256/AE_FiLM_embed_256__$i.json"
	python main_eval.py $val_json $i
done

