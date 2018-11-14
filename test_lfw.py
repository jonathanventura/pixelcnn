import tensorflow as tf
import numpy as np
from model import RGBModel
from tensorpack import *
from tensorpack.utils.viz import stack_patches
import sys
import cv2
from tqdm import trange

def get_lfw(subset,batch_size,shuffle,remainder):
    path = 'lfw_' + subset + '.h5'
    ds = dataflow.HDF5Data(path,['data'],shuffle=shuffle)
    augs = [
        imgaug.Resize(32)]
    ds = AugmentImageComponent(ds, augs)
    ds = BatchData(ds,batch_size,remainder=remainder)
    #ds = PrefetchDataZMQ(ds, 3)
    return ds

def sample_categorical(x):
    sample = [np.argmax(np.random.multinomial(1,p/(np.sum(p)+1e-5))) for p in x]
    return np.stack(sample)

pred_config = PredictConfig(
    model=RGBModel(),
    input_names=['image'],
    output_names=['probs'],
    session_init=SaverRestore(sys.argv[1])
)
predictor = OfflinePredictor(pred_config)

images = np.zeros((10,32,32,3),dtype='uint8')
for y in trange(32):
    for x in range(32):
        for c in range(3):
            probs, = predictor(images)
            sample = sample_categorical(probs[:,y,x,c])
            images[:,y,x,c] = sample
imout = stack_patches(images,2,5)
cv2.imwrite('generated.png',cv2.cvtColor(imout,cv2.COLOR_RGB2BGR))

ds_val = get_lfw('val',1,False,True)
ds_val.reset_state()
i = 0
for dp in ds_val:
    images[i] = dp[0]
    i = i + 1
    if i == len(images):
        break
images[:,16:,:,:] = 0
for y in trange(16,32):
    for x in range(32):
        for c in range(3):
            probs, = predictor(images)
            sample = sample_categorical(probs[:,y,x,c])
            images[:,y,x,c] = sample
imout = stack_patches(images,2,5)
cv2.imwrite('completed.png',cv2.cvtColor(imout,cv2.COLOR_RGB2BGR))
