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
flags.DEFINE_string("dataset_name", "mnist", "Dataset name (mnist,frey)")
flags.DEFINE_string("checkpoint_file", "./checkpoints/model.latest", "Path to checkpoint file")
flags.DEFINE_integer("num_block_cnn_filters", 16, "Number of channels in BlockCNN filters")
flags.DEFINE_integer("num_block_cnn_layers", 7, "Number of BlockCNN layers")
FLAGS = flags.FLAGS

def bernoulli(probs):
    ### sample binary vector according to bernoulli distribution
    ###
    return (np.random.sample(probs.shape)<=probs).astype('float32')

def gaussian(means):
    ### sample vector on [0,1] according to gaussian distribution centered at input
    ###
    x = means + np.random.standard_normal(means.shape)*0.01
    return np.clip(x,0,1)

def generate_images(model,image_height,image_width,labels,dist,sess):
    ### sequentially generate images
    images = np.zeros((len(labels),image_height,image_width,1),dtype='float32')
    for y in trange(image_height,desc='generating images'):
        for x in range(image_width):
            probs = model.predict(images,labels,sess)
            
            if dist == 'bernoulli':
                images[:,y,x] = bernoulli(probs[:,y,x])
            elif dist == 'gaussian':
                images[:,y,x] = gaussian(probs[:,y,x])
    return images

def autocomplete_images(model,images_in,labels,dist,sess,ystart=14):
    ### sequentially generate lower part of images
    images = np.copy(images_in)
    image_height = images.shape[1]
    image_width = images.shape[2]
    for y in trange(ystart,image_height,desc='generating images'):
        for x in range(image_width):
            probs = model.predict(images,labels,sess)
            
            if dist == 'bernoulli':
                images[:,y,x] = np.round(probs[:,y,x])
            elif dist == 'gaussian':
                images[:,y,x] = probs[:,y,x]
    return images

def main(_):
    # load data
    loader = DataLoader(dataset_name=FLAGS.dataset_name)
    FLAGS.image_height = loader.X_train.shape[1]
    FLAGS.image_width = loader.X_train.shape[2]
    if loader.y_train is not None:
        FLAGS.num_classes = loader.y_train.shape[1]
    else:
        FLAGS.num_classes = 1

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

        if FLAGS.num_classes > 1:
            labels = np.eye(FLAGS.num_classes,dtype='float32')
        else:
            labels = np.zeros((10,1),dtype='float32')
        images = generate_images(model,FLAGS.image_height,FLAGS.image_width,labels,loader.dist,sess)
        for i in range(len(images)):
            imsave('output/%02d_synt.png'%i,np.squeeze(images[i]*255.).astype('uint8'))

        if FLAGS.num_classes > 1:
            labels = loader.y_test[0:10]
        else:
            labels = np.zeros((10,1),dtype='float32')
        images = loader.X_test[0:10]
        auto_images = autocomplete_images(model,images,labels,loader.dist,sess)
        for i in range(len(images)):
            imsave('output/%02d_true.png'%i,np.squeeze(images[i]*255.).astype('uint8'))
            imsave('output/%02d_pred.png'%i,np.squeeze(auto_images[i]*255.).astype('uint8'))

if __name__ == '__main__':
    tf.app.run()
