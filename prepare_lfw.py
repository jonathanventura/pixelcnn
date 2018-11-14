import h5py as h5
import numpy as np
import cv2
import glob
import os
import sys

def load_image(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

if not os.path.exists('lfwcrop_color'):
    print('download lfwcrop_color.zip from  http://conradsanderson.id.au/lfwcrop/ and unzip it')
    sys.exit(1)

all_files = sorted(glob.glob('lfwcrop_color/faces/*.ppm'))

inds = list(range(len(all_files)))
val_inds = inds[::10]
train_inds = [ind for ind in inds if ind not in val_inds]

train_files = [all_files[ind] for ind in train_inds]
val_files = [all_files[ind] for ind in val_inds]

print(len(train_files),' training images')
print(len(val_files),' validation images')

train_images = np.stack([load_image(path) for path in train_files])
val_images = np.stack([load_image(path) for path in val_files])

with h5.File('lfw_train.h5','w') as f:
    f.create_dataset('data',data=train_images)
with h5.File('lfw_val.h5','w') as f:
    f.create_dataset('data',data=val_images)

