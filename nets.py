from __future__ import division
import tensorflow as tf
import numpy as np
from math import floor, ceil

def pixel_cnn_layer(vinput,hinput,filter_size,num_filters,layer_index,h=None):
    """Gated activation PixelCNN layer
       Code reference: https://github.com/dritchie/pixelCNN/blob/master/pixelCNN.lua
    """
    k = filter_size
    floork = int(floor(filter_size/2))
    ceilk = int(ceil(filter_size/2))
    
    # convolution for vertical stack
    vinput_padded = tf.pad(vinput, [[0,0],[ceilk,0],[floork,floork],[0,0]])
    vconv = tf.layers.conv2d(vinput_padded, 2*num_filters, [ceilk,k], name='vconv_%d'%layer_index,
        padding='valid', activation=None)
    vconv = vconv[:,:-1,:,:]
    
    # bias for vertical stack
    if h is not None:
        vbias = tf.layers.dense(h, 2*num_filters, name='vbias_%d'%layer_index,
            activation=None)
        vbias = tf.expand_dims(vbias,axis=1)
        vbias = tf.expand_dims(vbias,axis=1)
        vconv += vbias
    
    # apply separate activations
    vconv_tanh = tf.nn.tanh(vconv[:,:,:,:num_filters])
    vconv_sigmoid = tf.nn.sigmoid(vconv[:,:,:,num_filters:])
    
    # combine activations
    vconv = vconv_tanh * vconv_sigmoid
    
    # convolution for horizontal stack
    hinput_padded = tf.pad(hinput, [[0,0],[0,0],[ceilk,0],[0,0]])
    hconv = tf.layers.conv2d(hinput_padded, 2*num_filters, [1,ceilk], name='hconv_%d'%layer_index,
        padding='valid', activation=None)
    if layer_index==0:
        hconv = hconv[:,:,:-1,:]
    else:
        hconv = hconv[:,:,1:,:]

    # bias for horizontal stack
    if h is not None:
        hbias = tf.layers.dense(h, 2*num_filters, name='hbias_%d'%layer_index,
            activation=None)
        hbias = tf.expand_dims(hbias,axis=1)
        hbias = tf.expand_dims(hbias,axis=1)
        hconv += hbias
    
    # 1x1 transitional convolution for vstack
    vconv1 = tf.layers.conv2d(vconv, 2*num_filters, 1, name='vconv1_%d'%layer_index,
        padding='valid', activation=None)
    
    # add vstack to hstack
    hconv = hconv + vconv1
    
    # apply separate activations
    hconv_tanh = tf.nn.tanh(hconv[:,:,:,:num_filters])
    hconv_sigmoid = tf.nn.sigmoid(hconv[:,:,:,num_filters:])
    
    # combine activations
    hconv = hconv_tanh * hconv_sigmoid
    
    # residual connection in hstack
    hconv1 = tf.layers.conv2d(hconv, num_filters, 1, name='hconv1_%d'%layer_index,
        padding='valid', activation=None)
    hconv = hconv + hconv1
    
    return vconv, hconv

def pixel_cnn(inputs,num_filters,num_layers,h=None):
    """Builds PixelCNN graph.
    Args:
        inputs: input tensor (B,H,W,C)
        h: optional (B,K) tensor to condition the model on
    Returns:
        Predicted tensor
    """
    output_dim = inputs.get_shape()[3]
    with tf.variable_scope('pixel_cnn') as sc:
        vstack = inputs
        hstack = inputs
        
        # first layer: masked 7x7
        vstack, hstack = pixel_cnn_layer(vstack,hstack,7,num_filters,0,h=h)

        # next layers: masked 3x3
        for i in range(num_layers):
            vstack, hstack = pixel_cnn_layer(vstack,hstack,3,num_filters,i+1,h=h)
        
        # final layers
        x = tf.nn.relu(hstack)
        x = tf.layers.conv2d(x, num_filters, 1, name='conv',
            padding='valid', activation=tf.nn.relu)
        pred = tf.layers.conv2d(x, output_dim, 1, name='pred',
            padding='valid', activation=None)
        
    return pred

