import numpy as np
import tensorflow as tf
from pixelcnn import pixelcnn
from rgb_pixelcnn import rgb_pixelcnn
from tensorpack import *

class BinaryModel(ModelDesc):
    def inputs(self):
        return [tf.placeholder('float32',(None,None,None),name='image')]

    def build_graph(self, images):
        images = tf.expand_dims(images,axis=-1)

        # run PixelCNN model
        logits = pixelcnn(images*2.-1.,num_filters=32,num_layers=7,output_dim=1)
        probs = tf.sigmoid(tf.squeeze(logits,axis=-1),name='probs')

        # compute loss
        cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=images, logits=logits)
        per_image_loss = tf.reduce_sum(cross_entropy_loss,axis=[1,2])
        loss = tf.reduce_mean(per_image_loss,name='loss')

        # add summaries
        summary.add_moving_summary(loss)
        tf.summary.image('image',images)
        tf.summary.image('prediction',tf.expand_dims(probs,axis=-1))

        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        return tf.train.AdamOptimizer(lr)

class RGBModel(ModelDesc):
    def inputs(self):
        return [tf.placeholder('uint8',(None,None,None,3),name='image')]

    def build_graph(self, images):
        # run RGB PixelCNN model
        logits = rgb_pixelcnn(tf.cast(images,'float32')/255.*2.-1.,num_filters=32,num_layers=7,num_outputs=256)
        probs = tf.nn.softmax(logits,name='probs')
        pred = tf.cast(tf.argmax(logits,axis=-1),'uint8')

        # compute loss
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(images,'int32'), logits=logits)
        per_image_loss = tf.reduce_sum(cross_entropy_loss,axis=[1,2])
        loss = tf.reduce_mean(per_image_loss,name='loss')

        # add summaries
        summary.add_moving_summary(loss)
        tf.summary.image('image',images)
        tf.summary.image('prediction',pred)

        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        return tf.train.AdamOptimizer(lr)
