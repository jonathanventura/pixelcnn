#!/bin/sh
python test.py --dataset_name=cifar10 \
                --checkpoint_file=checkpoints/cifar10/model.latest \
                --num_block_cnn_filters=128 \
                --num_block_cnn_layers=15 \
                ;
