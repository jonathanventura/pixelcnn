#!/bin/sh
python test.py --dataset_name=frey \
                --checkpoint_file=checkpoints/frey/model.latest \
                --num_filters=16 \
                --num_layers=5 \
                ;
