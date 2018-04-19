#!/bin/sh
python test.py --dataset_name=mnist \
                --checkpoint_file=checkpoints/mnist/model.latest \
                --num_filters=16 \
                --num_layers=5 \
                ;
