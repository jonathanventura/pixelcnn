import argparse
import tensorflow as tf
import numpy as np
from model import BinaryModel
from tensorpack import *
from tensorpack.utils.viz import stack_patches
import sys
import cv2
from train_mnist import binarize, get_mnist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MNIST images')
    parser.add_argument('checkpoint', help='path to checkpoint')
    parser.add_argument('--complete', action='store_true', help='complete test images')
    args = parser.parse_args()

    # set up model configuration
    pred_config = PredictConfig(
        model=BinaryModel(),
        input_names=['image'],
        output_names=['probs'],
        session_init=SaverRestore(sys.argv[1])
    )
    predictor = OfflinePredictor(pred_config)

    # initialize images to zeros
    images = np.zeros((25,28,28),dtype='float32')

    # grab top half of test images if necessary
    if args.complete:
        print('getting test images')
        ds_test = get_mnist('test',1,False,True)
        i = 0
        for dp in ds_test:
            images[i] = dp[0]
            i = i + 1
            if i == len(images): break
        # zero out bottom half
        images[:,14:] = 0

    # run sequential prediction
    ystart = 14 if args.complete else 0
    for y in range(ystart,28):
        for x in range(28):
            probs, = predictor(images)
            sample = probs > np.random.sample(probs.shape)
            images[:,y,x] = sample[:,y,x]

    # generate output stack
    imout = stack_patches(images*255,5,5)
    cv2.imwrite('stack.png',imout)

