#!/bin/sh
python test.py --dataset_name=cifar10 \
                --checkpoint_file=checkpoints/cifar10/model.latest \
                --num_filters=128 \
                --num_layers=5 \
                ;
