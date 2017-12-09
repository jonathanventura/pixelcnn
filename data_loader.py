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
            
            self.has_labels = True
        else:
            raise ValueError('unknown dataset name %s'%dataset_name)

        if self.has_labels:
            train_dataset = tf.data.Dataset.from_tensor_slices((self.X_ph,self.y_ph))
            test_dataset = tf.data.Dataset.from_tensor_slices((self.X_ph,self.y_ph))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((self.X_ph))
            test_dataset = tf.data.Dataset.from_tensor_slices((self.X_ph))
        
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(self.batch_size)

        test_dataset = test_dataset.shuffle(buffer_size=1000)
        test_dataset = test_dataset.repeat()
        test_dataset = test_dataset.batch(self.batch_size)
        
        self.iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
        
        self.train_initializer = self.iterator.make_initializer(train_dataset)
        self.test_initializer = self.iterator.make_initializer(test_dataset)
    
    def setup_training(self,sess):
        if self.has_labels:
            sess.run(self.train_initializer,{ self.X_ph:self.X_train, self.y_ph:self.y_train })
        else:
            sess.run(self.train_initializer,{ self.X_ph:self.X_train })

    def setup_testing(self,sess):
        if self.has_labels:
            sess.run(self.test_initializer,{ self.X_ph:self.X_test, self.y_ph:self.y_test })
        else:
            sess.run(self.test_initializer,{ self.X_ph:self.X_test })

    def load_batch(self):
        """Load a batch of training/testing instances.
        """
        if self.has_labels:
            return self.iterator.get_next()
        else:
            return self.iterator.get_next(), None

