import tensorflow as tf
import numpy as np
from model import BinaryModel
from tensorpack import *
from tensorpack.utils.viz import stack_patches
import sys
import cv2

def binarize(x):
    return x > np.random.sample(x.shape)

pred_config = PredictConfig(
    model=BinaryModel(),
    input_names=['image'],
    output_names=['probs'],
    session_init=SaverRestore(sys.argv[1])
)
predictor = OfflinePredictor(pred_config)

images = np.zeros((10,28,28),dtype='float32')
for y in range(28):
    for x in range(28):
        probs, = predictor(images)
        sample = probs > np.random.sample(probs.shape)
        images[:,y,x] = sample[:,y,x]
imout = stack_patches(images,2,5)
cv2.imwrite('stack.png',imout)
