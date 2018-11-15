import tensorflow as tf
import numpy as np
from model import GrayscaleModel
from tensorpack import *
import glob

def get_frey(subset,batch_size,shuffle,remainder):
    path = 'frey_' + subset + '.h5'
    ds = dataflow.HDF5Data(path,['data'],shuffle=shuffle)
    ds = BatchData(ds,batch_size,remainder=remainder)
    ds = PrefetchDataZMQ(ds, 3)
    return ds

def get_data():
    ds_train = get_frey('train',16,True,False)
    ds_val = get_frey('val',16,True,False)
    return ds_train, ds_val

if __name__ == '__main__':
    ds_train, ds_val = get_data()

    logger.auto_set_dir()

    steps_per_epoch = len(ds_train)
    train_config = AutoResumeTrainConfig(
        model=GrayscaleModel(),
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_val,[ScalarStats(['loss'])]),
            MinSaver('validation_loss')
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=200,
    )
    launch_train_with_config(train_config, SimpleTrainer())

