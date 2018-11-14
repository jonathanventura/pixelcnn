from __future__ import division
import tensorflow as tf
import numpy as np
from math import floor, ceil

def _conv(inputs, num_outputs, filter_size, name):
    # relu activation
    #x = tf.nn.relu(inputs)
    x = inputs

    # get number of input channels
    num_channels_in = inputs.get_shape().as_list()[-1]

    # create filter weights
    weights = tf.get_variable(name + '_weights',
                              shape=filter_size + [num_channels_in,num_outputs],
                              initializer=tf.glorot_uniform_initializer())
    
    # create filter bias
    bias = tf.get_variable(name + '_bias',
                           shape=(num_outputs,),
                           initializer=tf.zeros_initializer())

    # apply filter and bias
    x = tf.nn.conv2d(x, weights, [1,1,1,1], 'VALID')
    x = tf.nn.bias_add(x, bias)

    return x

def _apply_activation(x):
    # split in half along channel axis
    x0,x1 = tf.split(x,2,axis=3)

    # apply separate activations
    x_tanh = tf.nn.tanh(x0)
    x_sigmoid = tf.nn.sigmoid(x1)
    
    # combine activations
    return x_tanh * x_sigmoid

def _gated_pixel_cnn_layer(vinput,hinput,filter_size,num_filters,layer_index):
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
    
    # kx1 convolution for horizontal stack
    hinput_padded = tf.pad(hinput, [[0,0],[0,0],[ceilk,0],[0,0]])
    hconv = _conv(hinput_padded, 2*num_filters, [1,ceilk], 'hconv_%d'%layer_index)
    if layer_index==0:
        hconv = hconv[:,:,:-1,:]
    else:
        hconv = hconv[:,:,1:,:]

    # 1x1 transitional convolution for vstack
    vconv1 = _conv(vconv, 2*num_filters, [1,1], 'vconv1_%d'%layer_index)
    
    # add vstack to hstack
    hconv = hconv + vconv1
    
    # apply activations
    vconv = _apply_activation(vconv)
    hconv = _apply_activation(hconv)
 
    # residual connection in hstack
    if layer_index > 0:
        hconv1 = _conv(hconv, num_filters, [1,1], 'hconv1_%d'%layer_index)
        hconv = hinput + hconv1
    
    return vconv, hconv

def gated_pixelcnn(inputs,num_filters,num_layers,num_outputs):
    """Builds Gated PixelCNN graph.
    Args:
        inputs: input tensor (B,H,W,C)
    Returns:
        Predicted tensor
    """
    with tf.variable_scope('pixel_cnn') as sc:
        vstack = inputs
        hstack = inputs
        
        # first layer: masked 7x7
        vstack, hstack = _gated_pixel_cnn_layer(vstack,hstack,7,num_filters,0)

        # next layers: masked 3x3
        for i in range(num_layers):
            vstack, hstack = _gated_pixel_cnn_layer(vstack,hstack,3,num_filters,i+1)
        
        # final layers
        x = _conv(tf.nn.relu(hstack), num_filters, [1,1], 'conv1')
        logits = _conv(tf.nn.relu(x), num_outputs, [1,1], 'logits')
        
    return logits

