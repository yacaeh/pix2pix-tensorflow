# -*- coding: utf-8 -*-
"""
@author: shoh4486
"""
import os
import tensorflow as tf
import numpy as np

def mkdir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)     
    except OSError:
        print('Cannot make the directory "{0}"'.format(directory))
    
def global_variables_list():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

def pixel_checker(image):
    if (np.min(image) < -1) or (np.max(image) > 1):
        raise ValueError('Pixel values should be in -1~1 range.')