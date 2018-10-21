from __future__ import division
import tensorflow as tf
import numpy as np
from math import floor, ceil

def _conv(inputs, num_outputs, filter_size, name):
    # relu activation
    x = tf.nn.relu(inputs)

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

def _hconv(inputs, num_outputs, filter_size, name, mask):
    # relu activation
    x = tf.nn.relu(inputs)

    # get number of input channels
    num_channels_in = inputs.get_shape().as_list()[-1]
    C = num_channels_in//3

    # create filter weights
    weights_shape = filter_size + [num_channels_in,num_outputs]
    maskR = np.ones(weights_shape,dtype='float32')
    maskG = np.ones(weights_shape,dtype='float32')
    maskB = np.ones(weights_shape,dtype='float32')
    if mask == 'A':
        maskR[:,-1,:,:] = 0
        maskG[:,-1,C:,:] = 0
        maskB[:,-1,2*C:,:] = 0
    elif mask == 'B':
        maskR[:,-1,C:,:] = 0
        maskG[:,-1,2*C:,:] = 0
    else:
        raise ValueError('unknown mask type: %s'%mask)

    weightsR = tf.get_variable(name + '_weightsR',
                              shape=weights_shape,
                              initializer=tf.glorot_uniform_initializer())
    weightsG = tf.get_variable(name + '_weightsG',
                              shape=weights_shape,
                              initializer=tf.glorot_uniform_initializer())
    weightsB = tf.get_variable(name + '_weightsB',
                              shape=weights_shape,
                              initializer=tf.glorot_uniform_initializer())
    weights = tf.concatenate([weightsR*maskR,weightsG*maskG,weightsB*maskB],axis=-1)
    
    # create filter bias
    bias = tf.get_variable(name + '_bias',
                           shape=(3*num_outputs,),
                           initializer=tf.zeros_initializer())

    # apply filter and bias
    x = tf.nn.conv2d(x, weights, [1,1,1,1], 'VALID')
    x = tf.nn.bias_add(x, bias)

    return x

def _pixel_cnn_layer(vinput,hinput,filter_size,num_filters,layer_index):
    """Simple PixelCNN layer with no gated activation"""
    k = filter_size
    floork = int(floor(filter_size/2))
    ceilk = int(ceil(filter_size/2))
    
    # kxk convolution for vertical stack
    vinput_padded = tf.pad(vinput, [[0,0],[ceilk,0],[floork,floork],[0,0]])
    vconv = _conv(vinput_padded, num_filters, [ceilk,k], 'vconv_%d'%layer_index)
    vconv = vconv[:,:-1,:,:]
    
    # kx1 convolution for horizontal stack
    hinput_padded = tf.pad(hinput, [[0,0],[0,0],[ceilk,0],[0,0]])
    mask = 'A' if layer_index == 0 else 'B'
    hconv = _hconv(hinput_padded, num_filters, [1,ceilk], 'hconv_%d'%layer_index, mask)
    hconv = hconv[:,:,1:,:]

    # 1x1 transitional convolution for vstack
    vconv1 = _conv(vconv, num_filters, [1,1], 'vconv1_%d'%layer_index)
    
    # add vstack to hstack
    hconv = hconv + vconv1
    
    # residual connection in hstack
    if layer_index > 0:
        hconv1 = _conv(hconv, num_filters, [1,1], 'hconv1_%d'%layer_index)
        hconv = hinput + hconv1
    
    return vconv, hconv

def rgb_pixelcnn(inputs,num_filters,num_layers):
    """Builds RGB PixelCNN graph.
    Args:
        inputs: input tensor (B,H,W,C)
    Returns:
        Predicted tensor
    """
    with tf.variable_scope('pixel_cnn') as sc:
        vstack = inputs
        hstack = inputs
        
        # first layer: masked 7x7
        vstack, hstack = _pixel_cnn_layer(vstack,hstack,7,num_filters,0)

        # next layers: masked 3x3
        for i in range(num_layers):
            vstack, hstack = _pixel_cnn_layer(vstack,hstack,3,num_filters,i+1)
        
        # final layers
        x = _conv(hstack, num_filters, [1,1], name='conv1')
        logits = _conv(x, 1, [1,1], name='logits')
        
    return logits

