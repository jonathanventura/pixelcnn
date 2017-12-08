from __future__ import division
import os
import random
import tensorflow as tf
import numpy as np

class DataLoader(object):
    def __init__(self, 
                 dataset_name=None,
                 batch_size=None):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        
        if dataset_name == 'mnist':
            (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()

            X_train = X_train.astype('float32')/255.
            X_train = np.expand_dims(X_train,axis=-1)

            y_train = tf.keras.utils.to_categorical(y_train,10)
            y_train = y_train.astype('float32')
            
            self.X_train = X_train
            self.y_train = y_train
            self.X_ph = tf.placeholder('float32',(None,28,28,1))
            self.y_ph = tf.placeholder('float32',(None,10))
            
            self.has_labels = True
        else:
            raise ValueError('unknown dataset name %s'%dataset_name)

        if self.has_labels:
            dataset = tf.data.Dataset.from_tensor_slices((self.X_ph,self.y_ph))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.X_ph))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        self.iterator = dataset.make_initializable_iterator()
    
    def setup_training(self,sess):
        if self.has_labels:
            sess.run(self.iterator.initializer,{ self.X_ph:self.X_train, self.y_ph:self.y_train })
        else:
            sess.run(self.iterator.initializer,{ self.X_ph:self.X_train })

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        if self.has_labels:
            return self.iterator.get_next()
        else:
            return self.iterator.get_next(), None

