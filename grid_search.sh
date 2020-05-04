#!/bin/bash

for hidden in 16 32 64 128 256
do
    for lr in 0.1 0.01 0.001 0.0001 0.00001
    do
        for lamda in 0.01 0.1 1 10 100
        do
            for dropout in 0.5 0.25 0.1
	    do
		echo "--hidden $hidden --lr $lr --lamda $lamda --dropout $dropout"
		python main.py --verbose 0 --hidden "$hidden" --lr "$lr" --lamda "$lamda" --dropout "$dropout" $@
	    done
        done
    done
done
