import argparse
import tensorflow as tf
import numpy as np
from model import RGBModel
from tensorpack import *
from tensorpack.utils.viz import stack_patches
import sys
import cv2
from tqdm import trange
from train_cifar10 import get_cifar10

def sample_categorical(x):
    sample = [np.argmax(np.random.multinomial(1,p/(np.sum(p)+1e-5))) for p in x]
    return np.stack(sample)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CIFAR-10 images')
    parser.add_argument('checkpoint', help='path to checkpoint')
    parser.add_argument('--complete', action='store_true', help='complete test images')
    args = parser.parse_args()

    # set up model configuration
    pred_config = PredictConfig(
        model=RGBModel(),
        input_names=['image'],
        output_names=['probs'],
        session_init=SaverRestore(args.checkpoint)
    )
    predictor = OfflinePredictor(pred_config)

    # initialize images to zeros
    images = np.zeros((25,32,32,3),dtype='uint8')

    # grab top half of test images if necessary
    if args.complete:
        print('getting test images')
        ds_test = get_cifar10('test',1,False,True)
        i = 0
        for dp in ds_test:
            images[i] = dp[0]
            i = i + 1
            if i == len(images): break
        # zero out bottom half
        images[:,16:] = 0

    # run sequential prediction
    ystart = 16 if args.complete else 0
    for y in trange(ystart,32):
        for x in range(32):
            for c in range(3):
                probs, = predictor(images)
                sample = sample_categorical(probs[:,y,x,c])
                images[:,y,x,c] = sample

    # generate output stack
    imout = stack_patches(images,5,5)
    cv2.imwrite('stack.png',imout)

