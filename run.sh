#!/bin/bash

sbatch --gres=gpu:1 --mem=20G --wrap="bash grid_search.sh --seed 1"
sbatch --gres=gpu:1 --mem=20G --wrap="bash grid_search.sh --seed 2"
sbatch --gres=gpu:1 --mem=20G --wrap="bash grid_search.sh --seed 3"
sbatch --gres=gpu:1 --mem=20G --wrap="bash grid_search.sh --seed 4"
sbatch --gres=gpu:1 --mem=20G --wrap="bash grid_search.sh --seed 5"


