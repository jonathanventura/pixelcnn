from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from PixelCNN import PixelCNN
from data_loader import DataLoader
import os
from tqdm import trange
from skimage.io import imsave

flags = tf.app.flags
flags.DEFINE_string("dataset_name", "mnist", "Dataset name (mnist)")
flags.DEFINE_string("checkpoint_file", "./checkpoints/", "Path to checkpoint file")
flags.DEFINE_integer("num_block_cnn_filters", 16, "Number of channels in BlockCNN filters")
flags.DEFINE_integer("num_block_cnn_layers", 7, "Number of BlockCNN layers")
FLAGS = flags.FLAGS

def binomial(probs):
    ### sample binary vector according to binomial distribution
    ###
    return (np.random.sample(probs.shape)<=probs).astype('float32')

def generate_images(model,image_height,image_width,labels,sess):
    ### sequentially generate images
    images = np.zeros((len(labels),image_height,image_width,1),dtype='float32')
    for y in trange(image_height,desc='generating images'):
        for x in range(image_width):
            probs = model.predict(images,labels,sess)
            
            # binomial sampling
            images[:,y,x] = binomial(probs[:,y,x])
    return images

def main(_):
    # load data
    loader = DataLoader(dataset_name=FLAGS.dataset_name)
    FLAGS.image_height = loader.X_train.shape[1]
    FLAGS.image_width = loader.X_train.shape[2]
    FLAGS.num_classes = loader.y_train.shape[1]

    # setup model for testing
    model = PixelCNN()
    model.setup_inference(FLAGS)

    if not os.path.exists('output'):
        os.makedirs('output')
    
    # load weights from file
    saver = tf.train.Saver([var for var in tf.trainable_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess,FLAGS.checkpoint_file)

        labels = np.eye(FLAGS.num_classes,dtype='float32')
        images = generate_images(model,FLAGS.image_height,FLAGS.image_width,labels,sess)
        for i in range(len(images)):
            imsave('output/%02d_image.png'%i,np.squeeze(images[i]*255.).astype('uint8'))

if __name__ == '__main__':
    tf.app.run()
