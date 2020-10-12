#!/bin/bash
NUM_ITERS=$1
if [ $# -lt 1 ]
then
	echo "Need to pass number of iterations."
fi

for i in $(seq "$NUM_ITERS")
do
	# Runs from 1 to $NUM_ITERS inclusive
	for context in {0..2}
	do 
		python main.py AE $context $i
		val_json="logs/ae/AE_"$context"_"$i".json"
                python main_eval.py $val_json $i
	done

done
