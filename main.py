# -*- coding: utf-8 -*-
"""
@author: shoh4486
tf.__version__ == '1.12.0' ~ '1.14.0'
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import os
import pprint
from model import Pix2pix
from utils import *

trial_num = 1

flags = tf.app.flags
flags.DEFINE_integer('height', 363, 'image height')
flags.DEFINE_integer('width', 298, 'image width')
flags.DEFINE_integer('in_channel', 3, 'input channel dimension')
flags.DEFINE_integer('out_channel', 1, 'output channel dimension')
flags.DEFINE_float('v_min', -10, 'minimum pixel value of raw data')
flags.DEFINE_float('v_max', 2000, 'maximum pixel value of raw data')
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
flags.DEFINE_string('pre_train_dir', os.path.join("./trials", "trial_{0}".format(23), "sess-{0}".format(1499)), 'when retraining, directory to restore. if none, just leave it.')
flags.DEFINE_integer('restart_epoch', 1500, 'restart epoch') 
flags.DEFINE_integer('re_end_epoch', 2000, 're-end epoch')
flags.DEFINE_boolean('eval_with_test_acc', True, 'True for test accuracies evaluation')
FLAGS = flags.FLAGS

pprint.pprint(flags.FLAGS.__flags)

mkdir(FLAGS.save_dir)
mkdir(os.path.join(FLAGS.save_dir, "test"))

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
    from data.data_preprocessing import inputs_train, inputs_train_, inputs_valid, gts_train, gts_train_, gts_valid
    data_col = [inputs_train, inputs_train_, inputs_valid, gts_train, gts_train_, gts_valid]
    for i in data_col:
        pixel_checker(i)
    
    if FLAGS.restore:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.pre_train_dir)
        FLAGS.start_epoch = FLAGS.restart_epoch
        FLAGS.end_epoch = FLAGS.re_end_epoch
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
    
else: # testing mode
    try:
        from data.test_data_preprocessing import inputs_test, gts_test
        test_data_col = [inputs_test, gts_test]
        for i in test_data_col:
            pixel_checker(i)
            
    except ImportError: # when gts_test is not given
        from data.test_data_preprocessing import inputs_test
        pixel_checker(inputs_test)
        
    if FLAGS.restore:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.pre_train_dir)
        if FLAGS.eval_with_test_acc:
            test_results = pix2pix.evaluation(
                                              inputs=inputs_test,
                                              gts=gts_test,
                                              with_h=False
                                              )
            G_c_test, MAE_test, MSE_test, R2_test, PSNR_test, SSIM_test = test_results
            np.savetxt(os.path.join(FLAGS.save_dir, "test", "MAE_test.txt"), MAE_test)
            np.savetxt(os.path.join(FLAGS.save_dir, "test", "MSE_test.txt"), MSE_test)
            np.savetxt(os.path.join(FLAGS.save_dir, "test", "R2_test.txt"), R2_test)
            np.savetxt(os.path.join(FLAGS.save_dir, "test", "PSNR_test.txt"), PSNR_test)
            np.savetxt(os.path.join(FLAGS.save_dir, "test", "SSIM_test.txt"), SSIM_test)
            
        else:
            G_c_test = pix2pix.evaluation(
                                          inputs=inputs_test,
                                          gts=None,
                                          with_h=False
                                          )
        for i in range(len(G_c_test)):
            if FLAGS.out_channel == 1:
                imsave(os.path.join(FLAGS.save_dir, "test", "test_result%d.png" % (i)), 0.5*G_c_test[i, :, :, 0] + 0.5, vmin=0.0, vmax=1.0, cmap=plt.cm.rainbow)
            else: # RGB or RGBA
                imsave(os.path.join(FLAGS.save_dir, "test", "test_result%d.png" % (i)), 0.5*G_c_test[i, :, :, :] + 0.5, vmin=0.0, vmax=1.0, cmap=plt.cm.rainbow)
    else:
        raise NotImplementedError('pretrained session must be restored.')