# -*- coding: utf-8 -*-
"""
@author: sio277(shoh4486@naver.com)
tf.__version__ == '1.12.0' ~ '1.14.0'
"""
import tensorflow as tf

def l_relu(inputs, alpha=0.2, name='leaky_relu'):
    """
    Leaky ReLU
    (Maas, A. L. et al., Rectifier nonlinearities imporve neural network acoustic models, Proc. icml. Vol.30. No.1. 2013)
    """
    return tf.maximum(inputs, alpha*inputs) # == tf.nn.leaky_relu(inputs, alpha)

def BN(inputs, is_training, name='batch_norm', momentum=0.99, center=True): 
    """
    Batch normalization
    (Ioffe, S. and Szegedy, C., Batch normalization: Accelerating deep network training by reducing internal covariate shift,
     arXiv preprint arXiv:1502.03167, 2015)
    
    Parameters
    inputs: [N, H, W, C]
    is_training: training mode check
    
    - Add tf.control_dependencies to the optimizer.
    """
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(inputs, momentum=momentum, epsilon=1e-5, center=center, training=is_training) 
    
def conv2d(inputs, FN, name='conv2d', FH=4, FW=4, sdy=1, sdx=1, padding='SAME', bias=True,
           weight_decay_lambda=None, truncated=False, stddev=0.02):
    """
    (standard) 2-D convolution
    
    Parameters
    inputs: [N, H, W, C]
    FN: filter number
    
    - filters: [FH, FW, C, FN]
    - outputs: [N, OH, OW, FN]
    """
    with tf.variable_scope(name):
        C = inputs.get_shape()[-1]
        initializer = tf.truncated_normal_initializer(stddev=stddev) if truncated else tf.random_normal_initializer(stddev=stddev)
             
        if not weight_decay_lambda:
            w = tf.get_variable(name='weight', shape=[FH, FW, C, FN], dtype=tf.float32, initializer=initializer)
        else:
            w = tf.get_variable(name='weight', shape=[FH, FW, C, FN], dtype=tf.float32, initializer=initializer,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay_lambda))
                
        conv = tf.nn.conv2d(inputs, w, strides=[1, sdy, sdx, 1], padding=padding)
        if not bias:
            return conv
        else:
            b = tf.get_variable(name='bias', shape=[FN], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv_ = tf.nn.bias_add(conv, b)
            return conv_
        
def t_conv2d(inputs, output_shape, name='t_conv2d', FH=4, FW=4, sdy=2, sdx=2, bias=True, 
             weight_decay_lambda=None, truncated=False, stddev=0.02):
    """
    2-D transposed convolution
    
    Parameters
    inputs: [N, H, W, C]
    output_shape: [N, OH, OW, FN]
    
    - filters: [FH, FW, FN, C] != filters.shape in conv2d 
    """
    with tf.variable_scope(name):
        FN = output_shape[-1]
        C = inputs.get_shape()[-1]
        initializer = tf.truncated_normal_initializer(stddev=stddev) if truncated else tf.random_normal_initializer(stddev=stddev)
        
        if not weight_decay_lambda:
            w = tf.get_variable(name='weight', shape=[FH, FW, FN, C], dtype=tf.float32, initializer=initializer) 
        else:
            w = tf.get_variable(name='weight', shape=[FH, FW, FN, C], dtype=tf.float32, initializer=initializer,
                                regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay_lambda))
            
        t_conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=[1, sdy, sdx, 1])
        if not bias:
            return t_conv
        else:
            b = tf.get_variable(name='bias', shape=[FN], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            t_conv_ = tf.nn.bias_add(t_conv, b)    
            return t_conv_