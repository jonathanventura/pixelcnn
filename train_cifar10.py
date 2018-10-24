import tensorflow as tf
import numpy as np
from model import RGBModel
from tensorpack import *

def get_cifar10(subset,batch_size,shuffle,remainder):
    ds = dataset.Cifar10(subset,shuffle=shuffle)
    ds = SelectComponent(ds,[0])
    ds = BatchData(ds,batch_size,remainder=remainder)
    return ds

def get_data():
    ds_train = get_cifar10('train',16,True,False)
    ds_test = get_cifar10('test',16,False,True)
    return ds_train, ds_test

if __name__ == '__main__':
    ds_train, ds_test = get_data()

    logger.auto_set_dir()

    steps_per_epoch = len(ds_train)
    train_config = AutoResumeTrainConfig(
        model=RGBModel(),
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

