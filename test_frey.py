import tensorflow as tf
import numpy as np
from model import GrayscaleModel
from tensorpack import *
from tensorpack.utils.viz import stack_patches
import sys
import cv2
from tqdm import trange

def get_frey(subset,batch_size,shuffle,remainder):
    path = 'frey_' + subset + '.h5'
    ds = dataflow.HDF5Data(path,['data'],shuffle=shuffle)
    ds = BatchData(ds,batch_size,remainder=remainder)
    ds = PrefetchDataZMQ(ds, 3)
    return ds

def sample_categorical(x):
    sample = [np.argmax(np.random.multinomial(1,p/(np.sum(p)+1e-5))) for p in x]
    return np.stack(sample)

pred_config = PredictConfig(
    model=GrayscaleModel(),
    input_names=['image'],
    output_names=['probs'],
    session_init=SaverRestore(sys.argv[1])
)
predictor = OfflinePredictor(pred_config)

images = np.zeros((10,28,20),dtype='uint8')
for y in trange(28):
    for x in range(20):
        probs, = predictor(images)
        sample = sample_categorical(probs[:,y,x])
        images[:,y,x] = sample
imout = stack_patches(images,2,5)
cv2.imwrite('generated.png',imout)

ds_val = get_frey('val',1,False,False)
ds_val.reset_state()
i = 0
for dp in ds_val:
    images[i] = dp[0]
    i = i + 1
    if i == len(images):
        break
images[:,14:,:] = 0
for y in trange(14,28):
    for x in range(20):
        probs, = predictor(images)
        sample = sample_categorical(probs[:,y,x])
        images[:,y,x] = sample
imout = stack_patches(images,2,5)
cv2.imwrite('completed.png',imout)

