#!/bin/sh
python test.py --dataset_name=frey \
                --checkpoint_file=checkpoints/frey/model.latest \
                --num_block_cnn_filters=16 \
                --num_block_cnn_layers=7 \
                ;
