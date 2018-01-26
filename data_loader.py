from __future__ import division
import os
import random
import tensorflow as tf
import numpy as np
from scipy.io import loadmat

def _random_batch(batch_size,X_in,y_in):
    inds = np.random.randint(len(X_in),size=batch_size)
    X = X_in[inds]
    if y_in is not None:
        y = y_in[inds]
    else:
        y = None
    return X, y

class DataLoader(object):
    def __init__(self, 
                 dataset_name=None,
                 batch_size=None):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        
        if dataset_name == 'mnist':
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

            X_train = X_train.astype('float32')/255.
            X_train = np.expand_dims(X_train,axis=-1)

            X_test = X_test.astype('float32')/255.
            X_test = np.expand_dims(X_test,axis=-1)
            
            # binarize data
            X_train = ( X_train > np.random.sample(X_train.shape) ).astype('float32')
            X_test = ( X_test > np.random.sample(X_test.shape) ).astype('float32')

            y_train = tf.keras.utils.to_categorical(y_train,10)
            y_train = y_train.astype('float32')

            y_test = tf.keras.utils.to_categorical(y_test,10)
            y_test = y_test.astype('float32')
            
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.X_ph = tf.placeholder('float32',(None,28,28,1))
            self.y_ph = tf.placeholder('float32',(None,10))
            
            self.dist = 'bernoulli'
        elif dataset_name == 'frey':
            path = tf.keras.utils.get_file('frey_rawface.mat','https://cs.nyu.edu/~roweis/data/frey_rawface.mat')
            data = np.transpose(loadmat(path)['ff'])
            X_train = np.reshape(data,(-1,28,20,1))

            X_train = X_train.astype('float32')/255.

            y_train = np.copy(X_train)

            self.X_train = X_train
            self.y_train = None
            self.X_test = X_train
            self.y_test = None
            self.X_ph = tf.placeholder('float32',(None,28,20,1))
            self.y_ph = None
            
            self.dist = 'gaussian'
        elif dataset_name == 'cifar10':
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

            X_train = X_train.astype('float32')/255.
            # convert to [-1 1]
            X_train = X_train*2.-1.

            X_test = X_test.astype('float32')/255.
            # convert to [-1 1]
            X_test = X_test*2.-1.
            
            y_train = tf.keras.utils.to_categorical(y_train,10)
            y_train = y_train.astype('float32')

            y_test = tf.keras.utils.to_categorical(y_test,10)
            y_test = y_test.astype('float32')
            
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.X_ph = tf.placeholder('float32',(self.batch_size,32,32,3))
            self.y_ph = tf.placeholder('float32',(self.batch_size,10))
            
            self.dist = 'logistic'
        else:
            raise ValueError('unknown dataset name %s'%dataset_name)

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        return _random_batch(self.batch_size,self.X_train,self.y_train)
    
    def load_test_batch(self):
        """Load a batch of testing instances.
        """
        return _random_batch(self.batch_size,self.X_test,self.y_test)

