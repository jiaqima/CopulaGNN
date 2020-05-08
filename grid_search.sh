#!/bin/bash

for hidden in 16 32 64 128 256
do
    for lr in 0.2 0.4 0.6 0.8 
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
