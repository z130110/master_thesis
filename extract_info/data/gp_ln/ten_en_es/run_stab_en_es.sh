#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=6000M
# we run on the gpu partition and we allocate 1 random gpu
#SBATCH -p gpu --gres=gpu:titanx:1
# We expect that our program should not run langer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=199:00:00
#SBATCH --output=log-%A.out

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
hostname
echo ${CUDA_VISIBLE_DEVICES}

python3 -u stab_gp_en_es.py
