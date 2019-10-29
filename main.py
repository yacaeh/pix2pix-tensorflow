# -*- coding: utf-8 -*-
"""
@author: sio277(shoh4486@naver.com)
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import os
import pprint
from model import Pix2pix
from utils import *
from data.data_preprocessing import inputs_train, inputs_train_, inputs_valid, inputs_test, gts_train, gts_train_, gts_valid, gts_test, v_min, v_max

trial_num = 24

flags = tf.app.flags
flags.DEFINE_integer('height', 363, 'image height')
flags.DEFINE_integer('width', 298, 'image width')
flags.DEFINE_integer('in_channel', 3, 'input channel dimension')
flags.DEFINE_integer('out_channel', 1, 'output channel dimension')
flags.DEFINE_float('v_min', v_min, 'minimum pixel value of the training data')
flags.DEFINE_float('v_max', v_max, 'maximum pixel value of the training data')
flags.DEFINE_integer('seed', 191015, 'seed number')
flags.DEFINE_float('loss_lambda', 100.0, 'L1 loss lambda')
flags.DEFINE_bool('LSGAN', False, 'applying LSGAN loss')
flags.DEFINE_float('weight_decay_lambda', 0.0, 'L2 weight decay lambda')
flags.DEFINE_bool('truncated', False, 'truncated weight distribution')
flags.DEFINE_string('optimizer', 'Adam', 'optimizer')
flags.DEFINE_string('save_dir', os.path.join("./trials", "trial_{0}".format(trial_num)), 'output saving directory')
flags.DEFINE_integer('gpu_num', 2, 'the number of GPUs')
flags.DEFINE_integer('batch_size_training', 2, 'batch size')
flags.DEFINE_float('lr_init', 1e-04, 'initial learning rate')
flags.DEFINE_integer('check_epoch', 5, 'check epoch')
flags.DEFINE_integer('start_epoch', 0, 'start epoch') 
flags.DEFINE_integer('end_epoch', 200, 'end epoch')
flags.DEFINE_bool('lr_decay', False, 'learning rate decay')
flags.DEFINE_boolean('train', False, 'True for training, False for evaluation')
flags.DEFINE_boolean('restore', True, 'True for retoring, False for raw training')
flags.DEFINE_string('pre_train_dir', os.path.join("./trials", "trial_{0}".format(23), "sess-{0}".format(1499)), 'when retraining, directory to restore')
flags.DEFINE_integer('restart_epoch', 1500, 'start epoch') 
flags.DEFINE_integer('reend_epoch', 2000, 'end epoch')
FLAGS = flags.FLAGS

pprint.pprint(flags.FLAGS.__flags)

create_directory(FLAGS.save_dir)
create_directory(os.path.join(FLAGS.save_dir, "sess"))
create_directory(os.path.join(FLAGS.save_dir, "test"))

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
sess = tf.Session(config=run_config)

pix2pix = Pix2pix(
                  sess=sess,
                  H_in=FLAGS.height,
                  W_in=FLAGS.width,
                  C_in=FLAGS.in_channel,   
                  C_out=FLAGS.out_channel,
                  v_min=FLAGS.v_min,
                  v_max=FLAGS.v_max,
                  seed=FLAGS.seed,
                  loss_lambda=FLAGS.loss_lambda,
                  LSGAN=FLAGS.LSGAN,
                  weight_decay_lambda=FLAGS.weight_decay_lambda,
                  truncated=FLAGS.truncated,
                  optimizer=FLAGS.optimizer,
                  save_dir=FLAGS.save_dir,
                  gpu_num=FLAGS.gpu_num
                  )

global_variables_list()

if FLAGS.train:
    if FLAGS.restore:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.pre_train_dir)
        FLAGS.start_epoch = FLAGS.restart_epoch
        FLAGS.end_epoch = FLAGS.reend_epoch
        pix2pix.train(
                      inputs=(inputs_train, inputs_train_, inputs_valid),
                      gts=(gts_train, gts_train_, gts_valid),
                      config=FLAGS
                      )
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(FLAGS.save_dir, "sess"), global_step=FLAGS.end_epoch-1)
    
    else:  
        pix2pix.train(
                      inputs=(inputs_train, inputs_train_, inputs_valid),
                      gts=(gts_train, gts_train_, gts_valid),
                      config=FLAGS
                      )
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(FLAGS.save_dir, "sess"), global_step=FLAGS.end_epoch-1)
    
    np.savetxt(os.path.join(FLAGS.save_dir, "MAE_train.txt"), pix2pix.MAE_train_vals)
    np.savetxt(os.path.join(FLAGS.save_dir, "MSE_train.txt"), pix2pix.MSE_train_vals)
    np.savetxt(os.path.join(FLAGS.save_dir, "R2_train.txt"), pix2pix.R2_train_vals)
    np.savetxt(os.path.join(FLAGS.save_dir, "PSNR_train.txt"), pix2pix.PSNR_train_vals)
    np.savetxt(os.path.join(FLAGS.save_dir, "SSIM_train.txt"), pix2pix.SSIM_train_vals)
    
    np.savetxt(os.path.join(FLAGS.save_dir, "MAE_valid.txt"), pix2pix.MAE_valid_vals)
    np.savetxt(os.path.join(FLAGS.save_dir, "MSE_valid.txt"), pix2pix.MSE_valid_vals)
    np.savetxt(os.path.join(FLAGS.save_dir, "R2_valid.txt"), pix2pix.R2_valid_vals)
    np.savetxt(os.path.join(FLAGS.save_dir, "PSNR_valid.txt"), pix2pix.PSNR_valid_vals)
    np.savetxt(os.path.join(FLAGS.save_dir, "SSIM_valid.txt"), pix2pix.SSIM_valid_vals)
    
else:
    if FLAGS.restore:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.pre_train_dir)
        test_result = pix2pix.evaluation(
                                         inputs=inputs_test,
                                         gts=gts_test,
                                         with_h=False
                                         )
        G_c, MAE, MSE, R2, PSNR, SSIM = test_result
        for i in range(len(G_c)):
            if FLAGS.out_channel == 1:
                imsave(os.path.join(FLAGS.save_dir, "test", "test_result_%d.png" % (i)), 0.5*G_c[i, :, :, 0] + 0.5, vmin=0.0, vmax=1.0)
            else: # RGB or RGBA
                imsave(os.path.join(FLAGS.save_dir, "test", "test_result_%d.png" % (i)), 0.5*G_c[i, :, :, :] + 0.5, vmin=0.0, vmax=1.0)
    else:
        raise NotImplementedError('pretrained session must be restored.')