# -*- coding: utf-8 -*-
"""
@author: shoh4486
tf.__version__ == '1.12.0' ~ '1.14.0'
"""
import tensorflow as tf
import numpy as np
import os
import pprint
from model import Pix2pix
from utils import *

flags = tf.app.flags
# for model class instantiation
flags.DEFINE_integer('height', 363, 'image height')
flags.DEFINE_integer('width', 298, 'image width')
flags.DEFINE_integer('in_channel', 3, 'the number of input channels')
flags.DEFINE_integer('out_channel', 1, 'the number of output channels')
flags.DEFINE_float('v_min', -10, 'minimum pixel value of the ground truth (before normalization)')
flags.DEFINE_float('v_max', 2000, 'maximum pixel value of the ground truth (before normalization)')
# FLAGS.v_min, FLAGS.v_max: to calculate MAE and MSE errors by reflecting grount truths' original range
flags.DEFINE_integer('seed', 1, 'seed number')
flags.DEFINE_float('loss_lambda', 100.0, 'L1 loss lambda')
flags.DEFINE_bool('LSGAN', False, 'applying LSGAN loss')
flags.DEFINE_float('weight_decay_lambda', 0.0, 'L2 weight decay lambda')
flags.DEFINE_string('optimizer', 'Adam', 'optimizer')
flags.DEFINE_list('gpu_alloc', ['1', '2'], 'specifying which GPU(s) to be used; set to 0 to use only cpu')
# Registers a flag whose value is a comma-separated list of strings, e.g. ['1', '2'].
# e.g. set --gpu_alloc=1,2 if to use the first and the second GPUs.
# Note: the order of elements in FLAGS.gpu_alloc should be correctly inserted.
# if it is 2,1, the first GPU is assigned to '/device:GPU:1' and the second GPU to '/device:GPU:0'.
# if it is 3,4, the third GPU is assgined to '/device:GPU:0' and the fourth GPU to '/device:GPU:1'.
#
flags.DEFINE_integer('trial_num', 1, 'trial number')
flags.DEFINE_integer('batch_size_training', 2, 'batch size')
flags.DEFINE_float('lr_init', 1e-04, 'initial learning rate')
flags.DEFINE_bool('lr_decay', False, 'applying learning rate decay')
#
flags.DEFINE_boolean('train', True, 'True for training, False for testing')
flags.DEFINE_boolean('restore', False, 'True for restoring, False for raw training')
flags.DEFINE_integer('start_epoch', 0, 'start epoch') 
flags.DEFINE_integer('end_epoch', 200, 'end epoch')
flags.DEFINE_integer('check_epoch', 5, 'check epoch')
# if not restoring, do not concern below flags.
flags.DEFINE_integer('restore_trial_num', 1, 'directory number of the pretrained model')
flags.DEFINE_integer('restore_sess_num', 199, 'sess number of the pretrained model')
flags.DEFINE_boolean('eval_with_test_acc', True, 'True for test accuracies evaluation')
FLAGS = flags.FLAGS

def main(_):
    flags.DEFINE_string('save_dir', os.path.join("./trials", "trial_{0}".format(FLAGS.trial_num)), 
                        'output saving directory')
    pprint.pprint(flags.FLAGS.__flags)
    
    mkdir(FLAGS.save_dir)
    mkdir(os.path.join(FLAGS.save_dir, "test"))
    mkdir(os.path.join(FLAGS.save_dir, "loss_acc"))
    
    if FLAGS.gpu_alloc == ['0']:
        run_config = tf.ConfigProto(device_count={'GPU': 0}) 
        # even if there are GPUs, they will be ignored.
        sess = tf.Session(config=run_config)
    else:
        assert '0' not in FLAGS.gpu_alloc      
        visible_device_list = ','.join([str(int(i) - 1) for i in FLAGS.gpu_alloc])
        # If FLAGS.gpu_alloc == ['1', '2'], it is converted to '0,1'. GPU number starts from 0.        
        # Method1: Specify to-be-used GPUs in tf.GPUOptions. Other GPUs will be blinded.
        gpu_options = tf.GPUOptions(
            allow_growth=True, 
            visible_device_list=visible_device_list
            )
        # Method2: Specify to-be-used GPUs in CUDA. 
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # os.environ['CUDA_VISIBLE_DEVICES'] = visible_device_list
        # gpu_options = tf.GPUOptions(allow_growth=True)
        run_config = tf.ConfigProto(gpu_options=gpu_options)
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
                      optimizer=FLAGS.optimizer,
                      gpu_alloc=FLAGS.gpu_alloc
                      )
    
    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
    if FLAGS.train:
        from data.data_preprocessing import inputs_train, inputs_train_, \
        inputs_valid, gts_train, gts_train_, gts_valid
        data_col = [inputs_train, inputs_train_, inputs_valid, gts_train, gts_train_, gts_valid]
        for i in data_col:
            pixel_checker(i)
        
        if FLAGS.restore:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join("./trials", "trial_{0}".format(FLAGS.restore_trial_num), 
                                             "sess-{0}".format(FLAGS.restore_sess_num)))
            pix2pix.train(
                          inputs=(inputs_train, inputs_train_, inputs_valid),
                          gts=(gts_train, gts_train_, gts_valid),
                          config=FLAGS
                          )       
        else:  
            pix2pix.train(
                          inputs=(inputs_train, inputs_train_, inputs_valid),
                          gts=(gts_train, gts_train_, gts_valid),
                          config=FLAGS
                          )
            
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(FLAGS.save_dir, "sess"), global_step=FLAGS.end_epoch-1)
        
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "MAE_train.txt"), 
                   pix2pix.MAE_train_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "MSE_train.txt"), 
                   pix2pix.MSE_train_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "R2_train.txt"), 
                   pix2pix.R2_train_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "PSNR_train.txt"), 
                   pix2pix.PSNR_train_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "SSIM_train.txt"), 
                   pix2pix.SSIM_train_vals)
        
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "MAE_valid.txt"), 
                   pix2pix.MAE_valid_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "MSE_valid.txt"), 
                   pix2pix.MSE_valid_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "R2_valid.txt"), 
                   pix2pix.R2_valid_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "PSNR_valid.txt"), 
                   pix2pix.PSNR_valid_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "loss_acc", "SSIM_valid.txt"), 
                   pix2pix.SSIM_valid_vals)
        
    else: # testing mode
        try:
            from data.test_data_preprocessing import inputs_test, gts_test
        except ImportError: # when gts_test is not given
            from data.test_data_preprocessing import inputs_test
        else:
            pixel_checker(gts_test)
        finally:
            pixel_checker(inputs_test)
            
        if FLAGS.restore:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join("./trials", "trial_{0}".format(FLAGS.restore_trial_num), 
                                             "sess-{0}".format(FLAGS.restore_sess_num)))
            if FLAGS.eval_with_test_acc:
                test_results = pix2pix.evaluation(
                                                  inputs=inputs_test,
                                                  gts=gts_test,
                                                  is_training=False,
                                                  with_h=False
                                                  )
                G_c_test, MAE_test, MSE_test, R2_test, PSNR_test, SSIM_test = test_results
                np.savetxt(os.path.join(FLAGS.save_dir, "test", "MAE_test.txt"), 
                           MAE_test)
                np.savetxt(os.path.join(FLAGS.save_dir, "test", "MSE_test.txt"), 
                           MSE_test)
                np.savetxt(os.path.join(FLAGS.save_dir, "test", "R2_test.txt"), 
                           R2_test)
                np.savetxt(os.path.join(FLAGS.save_dir, "test", "PSNR_test.txt"), 
                           PSNR_test)
                np.savetxt(os.path.join(FLAGS.save_dir, "test", "SSIM_test.txt"), 
                           SSIM_test)
                
            else:
                G_c_test = pix2pix.evaluation(
                                              inputs=inputs_test,
                                              gts=None,
                                              is_training=False,
                                              with_h=False
                                              )
            # denormalize to the original range (from -1~1 to v_min~v_max)
            G_c_test = pix2pix._alpha*G_c_test + pix2pix._beta      
            for i in range(len(G_c_test)):
                for c in range(FLAGS.out_channel):
                    np.savetxt(os.path.join(FLAGS.save_dir, "test", "test_result%d_channel%d.txt" % (i, c)), 
                               G_c_test[i, :, :, c])
        else:
            raise NotImplementedError('pretrained session must be restored.')
            
if __name__ == '__main__':
    tf.app.run()