#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Mon Dec 18 2017 
@Time    : 22:47:13
@File    : train.py
@Author  : alpha
"""


import keras.backend as K
from model import CodeDetector

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

class Config:
    start_lr = 0.001
    train_batch_size = 32
    val_batch_size = 32
    epoches = 200
    save_model_name = 'model'


tfconf = tf.ConfigProto()
tfconf.gpu_options.allow_growth = True

K.set_session(tf.Session(config=tfconf))

cfg = Config()
myocr = CodeDetector(cfg)
myocr.show_model()
myocr.train()

K.clear_session()