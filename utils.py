# -*- coding: utf-8 -*-
"""
@author: sio277(shoh4486@naver.com)
"""
import os
import tensorflow as tf
import numpy as np

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    
def global_variables_list():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

def pixel_checker(image):
    if (np.min(image) < -1) or (np.max(image) > 1):
        raise ValueError('Pixel value should be in -1~1 range.')