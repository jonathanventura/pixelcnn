import h5py as h5
import numpy as np
import cv2
import glob
import os
import sys
from scipy.io import loadmat

if not os.path.exists('frey_rawface.mat'):
    print('download https://cs.nyu.edu/~roweis/data/frey_rawface.mat')
    sys.exit(1)

data = np.transpose(loadmat('frey_rawface.mat')['ff'])
images = np.reshape(data,(-1,28,20))

idxs = list(range(len(images)))
val_idxs = idxs[::10]
train_idxs = [idx for idx in idxs if not idx in val_idxs]
train_images = images[train_idxs]
val_images = images[val_idxs]

with h5.File('frey_train.h5','w') as f:
    f.create_dataset('data',data=train_images)
with h5.File('frey_val.h5','w') as f:
    f.create_dataset('data',data=val_images)

