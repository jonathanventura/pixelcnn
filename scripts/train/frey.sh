#!/bin/sh
mkdir -p checkpoints/frey ;
rm checkpoints/frey/* ;
python train.py --dataset_name=frey \
                --checkpoint_dir=checkpoints/frey \
                --num_filters=16 \
                --num_layers=5 \
                ;
