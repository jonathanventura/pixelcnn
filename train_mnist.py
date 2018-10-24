import tensorflow as tf
import numpy as np
from model import BinaryModel
from tensorpack import *

def binarize(x):
    return x > np.random.sample(x.shape)

def get_mnist(subset,batch_size,shuffle,remainder):
    ds = dataset.Mnist(subset,shuffle=shuffle)
    ds = SelectComponent(ds,[0])
    ds = MapDataComponent(ds,binarize)
    ds = BatchData(ds,batch_size,remainder=remainder)
    return ds

def get_data():
    ds_train = get_mnist('train',16,True,False)
    ds_test = get_mnist('test',16,False,True)
    return ds_train, ds_test

if __name__ == '__main__':
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

