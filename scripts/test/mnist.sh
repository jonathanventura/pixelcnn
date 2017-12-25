#!/bin/sh
python test.py --dataset_name=mnist \
                --checkpoint_file=checkpoints/mnist/model.latest \
                --num_block_cnn_filters=16 \
                --num_block_cnn_layers=7 \
                ;
