import tensorflow as tf
import numpy as np
from model import BinaryModel
from tensorpack import *

def get_data():
    ds_train = dataset.Mnist('train',shuffle=True)
    ds_train = SelectComponent(ds_train,[0])
    ds_train = BatchData(ds_train,16,remainder=False)

    ds_test = dataset.Mnist('test',shuffle=False)
    ds_test = SelectComponent(ds_test,[0])
    ds_test = BatchData(ds_test,16,remainder=True)
    
    return ds_train, ds_test

ds_train, ds_test = get_data()

logger.auto_set_dir()

steps_per_epoch = len(ds_train)
train_config = AutoResumeTrainConfig(
    model=BinaryModel(),
    data=FeedInput(ds_train),
    callbacks=[
        ModelSaver(),
        MinSaver('validation_loss'),
        InferenceRunner(ds_test,[ScalarStats(['loss'])])
    ],
    steps_per_epoch=steps_per_epoch,
    max_epoch=200,
)
launch_train_with_config(train_config, SimpleTrainer())

