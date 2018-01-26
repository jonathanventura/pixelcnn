#!/bin/sh
mkdir -p checkpoints/cifar10 ;
rm checkpoints/cifar10/* ;
python train.py --dataset_name=cifar10 \
                --checkpoint_dir=checkpoints/cifar10 \
                --num_filters=128 \
                --num_layers=15 \
                ;
