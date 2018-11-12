import tensorflow as tf
import numpy as np
from model import RGBModel
from tensorpack import *
import glob

def get_lfw(subset,batch_size,shuffle,remainder):
    path = 'lfw_' + subset + '.h5'
    ds = dataflow.HDF5Data(path,['data'],shuffle=shuffle)
    augs = [
        imgaug.Resize(32)]
    ds = AugmentImageComponent(ds, augs)
    ds = BatchData(ds,batch_size,remainder=remainder)
    #ds = PrefetchDataZMQ(ds, 3)
    return ds

def get_data():
    ds_train = get_lfw('train',16,True,False)
    ds_val = get_lfw('val',16,True,False)
    return ds_train, ds_val

if __name__ == '__main__':
    ds_train, ds_val = get_data()

    logger.auto_set_dir()

    steps_per_epoch = len(ds_train)
    train_config = AutoResumeTrainConfig(
        model=RGBModel(),
        data=FeedInput(ds_train),
        callbacks=[
            ModelSaver(),
            MinSaver('validation_loss'),
            InferenceRunner(ds_val,[ScalarStats(['loss'])])
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=200,
    )
    launch_train_with_config(train_config, SimpleTrainer())

