#!/bin/sh
mkdir -p checkpoints/mnist ;
rm checkpoints/mnist/* ;
python train.py --dataset_name=mnist \
                --checkpoint_dir=checkpoints/mnist \
                --num_block_cnn_filters=16 \
                --num_block_cnn_layers=7 \
                ;
