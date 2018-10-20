from __future__ import division
import tensorflow as tf
import numpy as np
from math import floor, ceil

def _conv(inputs, num_outputs, filter_size, name):
    # relu activation
    x = tf.nn.relu(inputs)

    # convolution
    x = tf.layers.conv2d(x, num_outputs, filter_size,
            padding='valid', activation=None, name=name)

    return x

def _get_bias(h,num_filters,name):
    bias = tf.layers.dense(h, 2*num_filters, name=name,
        activation=None, use_bias=False)
    bias = tf.expand_dims(bias,axis=1)
    bias = tf.expand_dims(bias,axis=1)
    return bias

def _apply_activation(x,num_filters):
    # apply separate activations
    x_tanh = tf.nn.tanh(x[:,:,:,:num_filters])
    x_sigmoid = tf.nn.sigmoid(x[:,:,:,num_filters:])
    
    # combine activations
    return x_tanh * x_sigmoid

def _gated_pixel_cnn_layer(vinput,hinput,filter_size,num_filters,layer_index,h=None):
    """Gated activation PixelCNN layer
       Paper reference: https://arxiv.org/pdf/1606.05328.pdf
       Code reference: https://github.com/dritchie/pixelCNN/blob/master/pixelCNN.lua
    """
    k = filter_size
    floork = int(floor(filter_size/2))
    ceilk = int(ceil(filter_size/2))
    
    # kxk convolution for vertical stack
    vinput_padded = tf.pad(vinput, [[0,0],[ceilk,0],[floork,floork],[0,0]])
    vconv = _conv(vinput_padded, 2*num_filters, [ceilk,k], 'vconv_%d'%layer_index)
    vconv = vconv[:,:-1,:,:]
    
    # bias for vertical stack
    if h is not None:
        vconv += _get_bias(h,num_filters,'vbias_%d'%layer_index)

    # kx1 convolution for horizontal stack
    hinput_padded = tf.pad(hinput, [[0,0],[0,0],[ceilk,0],[0,0]])
    hconv = _conv(hinput_padded, 2*num_filters, [1,ceilk], 'hconv_%d'%layer_index)
    if layer_index==0:
        hconv = hconv[:,:,:-1,:]
    else:
        hconv = hconv[:,:,1:,:]

    # bias for horizontal stack
    if h is not None:
        hconv += _get_bias(h,num_filters,'hbias_%d'%layer_index)
    
    # 1x1 transitional convolution for vstack
    vconv1 = _conv(vconv, 2*num_filters, 1, 'vconv1_%d'%layer_index)
    
    # add vstack to hstack
    hconv = hconv + vconv1
    
    # apply activations
    vconv = _apply_activation(vconv,num_filters)
    hconv = _apply_activation(hconv,num_filters)
 
    # residual connection in hstack
    if layer_index > 0:
        hconv1 = _conv(hconv, num_filters, 1, 'hconv1_%d'%layer_index)
        hconv = hinput + hconv1
    
    return vconv, hconv

def gated_pixelcnn(inputs,num_filters,num_layers,output_dim,h=None):
    """Builds Gated PixelCNN graph.
    Args:
        inputs: input tensor (B,H,W,C)
        h: optional (B,K) tensor to condition the model on
    Returns:
        Predicted tensor
    """
    with tf.variable_scope('pixel_cnn') as sc:
        vstack = inputs
        hstack = inputs
        
        # first layer: masked 7x7
        vstack, hstack = _gated_pixel_cnn_layer(vstack,hstack,7,num_filters,0,h=h)

        # next layers: masked 3x3
        for i in range(num_layers):
            vstack, hstack = _gated_pixel_cnn_layer(vstack,hstack,3,num_filters,i+1,h=h)
        
        # final layers
        x = _conv(hstack, num_filters, 1, 'conv')
        logits = _conv(x, output_dim, 1, 'pred')
        
    return logits

