#!/bin/bash

for hidden in 16 64 256
do
    for lr in 0.01 0.1 0.2 0.4
    do
        for temperature in 1 2 5 
        do
            for dropout in 0.5 0.25 0.1
	    do
		echo "--hidden $hidden --lr $lr --temperature $temperature --dropout $dropout"
		python main.py --model regressioncgcn --verbose 0 --hidden "$hidden" --lr "$lr" --temperature "$temperature" --dropout "$dropout" $@
	    done
        done
    done
done
