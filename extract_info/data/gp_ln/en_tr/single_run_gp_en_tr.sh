#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=6000M
# we run on the gpu partition and we allocate 1 random gpu
#SBATCH -p gpu --gres=gpu:titanx:1
# We expect that our program should not run langer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=190:00:00
#SBATCH --output=log-%A.out

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
hostname
echo ${CUDA_VISIBLE_DEVICES}

python3 -u ../unsupervised.py --layer_norm True --n_epochs 100 --grad_lambda 0.5 --normalize_embeddings "center" --map_optimizer "adam,lr=0.0005" --dis_optimizer "adam,lr=0.0005" --src_lang "en" --tgt_lang "tr" --src_emb "../../data/wiki.en.vec" --tgt_emb "../../data/wiki.tr.vec" 
