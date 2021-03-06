import tensorflow as tf
import numpy as np
from model import RGBModel
from tensorpack import *
import glob

def get_celeba(subset,batch_size,shuffle,remainder):
    all_files = sorted(glob.glob('img_align_celeba/*.jpg'))
    inds = list(range(len(all_files)))
    val_inds = inds[::10]
    train_inds = [ind for ind in inds if ind not in val_inds]
    if subset == 'train':
        file_list = [all_files[ind] for ind in train_inds]
    elif subset == 'val':
        file_list = [all_files[ind] for ind in val_inds]
    else:
        raise ValueError('unknown subset %s'%subset)

    ds = ImageFromFile(file_list, channel=3, shuffle=shuffle)
    augs = [
        imgaug.CenterCrop(160),
        imgaug.Resize(32)]
    ds = AugmentImageComponent(ds, augs)
    ds = BatchData(ds,batch_size,remainder=remainder)
    ds = PrefetchDataZMQ(ds, 3)
    return ds

def get_data():
    ds_train = get_celeba('train',16,True,False)
    ds_val = get_celeba('val',16,False,True)
    return ds_train, ds_val

if __name__ == '__main__':
    ds_train, ds_val = get_data()

    logger.auto_set_dir()

    steps_per_epoch = len(ds_train)
    train_config = AutoResumeTrainConfig(
        model=RGBModel(),
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

