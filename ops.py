# -*- coding: utf-8 -*-
"""
@author: shoh4486
tf.__version__ == '1.12.0' ~ '1.14.0'
"""
import tensorflow as tf

def l_relu(inputs, alpha=0.2, name='leaky_relu'):
    """
    Leaky ReLU
    (Maas, A. L. et al., Rectifier nonlinearities imporve neural network acoustic models, Proc. icml. Vol.30. No.1. 2013)
    """
    return tf.maximum(inputs, alpha*inputs) # == tf.nn.leaky_relu(inputs, alpha)

class BN:
    """
    Batch normalization
    (Ioffe, S. and Szegedy, C., Batch normalization: Accelerating deep network training by reducing internal covariate shift,
     arXiv preprint arXiv:1502.03167, 2015)
    
    - Add tf.control_dependencies to the optimizer.
    """
    def __init__(self, momentum=0.99, epsilon=1e-5, center=True):
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        
    def __call__(self, inputs, is_training, name):
        """
        Parameters
        inputs: [N, H, W, C]
        is_training: training mode check
        """
        with tf.variable_scope(name):
            return tf.layers.batch_normalization(inputs, momentum=self.momentum, epsilon=self.epsilon, 
                                                 center=self.center, training=is_training)

class Conv2D:
    """
    (standard) 2-D convolution
    """
    def __init__(self, FH=4, FW=4, weight_decay_lambda=None, truncated=False, stddev=0.02, bias=True):
        self.FH, self.FW = FH, FW       
        self.weight_decay_lambda = weight_decay_lambda
        self.truncated = truncated
        self.stddev = stddev
        self.bias = bias
        
    def __call__(self, inputs, FN, s=1, name='conv2d', padding='SAME'):
        """
        Parameters
        inputs: [N, H, W, C]
        FN: filter number
        s: convolutional stride (sdy=sdx=s)
        
        - filters: [FH, FW, C, FN]
        - outputs: [N, OH, OW, FN]
        """
        with tf.variable_scope(name):
            sdy, sdx = s, s
            C = inputs.get_shape()[-1]
            initializer = tf.truncated_normal_initializer(stddev=self.stddev) if self.truncated else tf.random_normal_initializer(stddev=self.stddev)
                 
            if not self.weight_decay_lambda:
                w = tf.get_variable(name='weight', shape=[self.FH, self.FW, C, FN], dtype=tf.float32, initializer=initializer)
            else:
                w = tf.get_variable(name='weight', shape=[self.FH, self.FW, C, FN], dtype=tf.float32, initializer=initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay_lambda))
                    
            conv = tf.nn.conv2d(inputs, w, strides=[1, sdy, sdx, 1], padding=padding)
            if not self.bias:
                return conv
            else:
                b = tf.get_variable(name='bias', shape=[FN], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                conv_ = tf.nn.bias_add(conv, b)
                return conv_
        
class TConv2D:
    """
    2-D transposed convolution
    """
    def __init__(self, FH=4, FW=4, weight_decay_lambda=None, truncated=False, stddev=0.02, bias=True):
        self.FH, self.FW = FH, FW     
        self.weight_decay_lambda = weight_decay_lambda
        self.truncated = truncated
        self.stddev = stddev
        self.bias = bias
        
    def __call__(self, inputs, output_shape, s=2, name='t_conv2d'):
        """
        Parameters
        inputs: [N, H, W, C]
        output_shape: [N, OH, OW, FN]
        s: convolutional stride (sdy=sdx=s)
        
        - filters: [FH, FW, FN, C] != filters.shape in conv2d 
        """
        with tf.variable_scope(name):
            sdy, sdx = s, s
            FN = output_shape[-1]
            C = inputs.get_shape()[-1]
            initializer = tf.truncated_normal_initializer(stddev=self.stddev) if self.truncated else tf.random_normal_initializer(stddev=self.stddev)
            
            if not self.weight_decay_lambda:
                w = tf.get_variable(name='weight', shape=[self.FH, self.FW, FN, C], dtype=tf.float32, initializer=initializer) 
            else:
                w = tf.get_variable(name='weight', shape=[self.FH, self.FW, FN, C], dtype=tf.float32, initializer=initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay_lambda))
                
            t_conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=[1, sdy, sdx, 1])
            if not self.bias:
                return t_conv
            else:
                b = tf.get_variable(name='bias', shape=[FN], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                t_conv_ = tf.nn.bias_add(t_conv, b)    
                return t_conv_