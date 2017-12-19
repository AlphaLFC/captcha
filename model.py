#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Mon Dec 18 2017 
@Time    : 21:42:55
@File    : model.py
@Author  : alpha
"""

import keras

version = keras.__version__
assert int(version[:version.find('.')]) >= 2, 'Main version of keras should be >=2.'

import keras.backend as K

assert K.image_dim_ordering() == 'tf', 'Image array should be in tf mode.'

import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from keras.callbacks import *

from dataloader import datagen, decode
from dataloader import CHARS, MAX_LEN


def ctc_loss(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_accuracy(y_true, y_pred, max_len=MAX_LEN):
    labels = y_true[:, 2:]
    input_length = y_true[:, 0]
    decoded = K.ctc_decode(y_pred, input_length)[0][0]
    cmp = K.cast(K.equal(labels, decoded), dtype='float')
    return K.cast(K.equal(K.sum(cmp, axis=-1), max_len), dtype='float')


def normalize(x_tensor):
    x_tensor = K.cast(x_tensor, dtype='float32')
    normed = -0.5 + x_tensor / 255.0
    return normed


def res_block(inlayer, n_channel, strides=(1, 1), regularizer=None, name=None):
    assert name is not None, 'Please specify a name.'
    conv1a = Conv2D(n_channel, (3, 3), strides=strides, padding='same', activation='relu',
                    kernel_regularizer=regularizer, name=name + '_conv1a')(inlayer)
    conv1a_bn = BatchNormalization(name=name + '_conv1a_bn')(conv1a)
    conv2a = Conv2D(n_channel, (3, 3), strides=(1, 1), padding='same', activation='relu',
                    kernel_regularizer=regularizer, name=name + '_conv2a')(conv1a_bn)
    conv2a_bn = BatchNormalization(name=name + '_conv2a_bn')(conv2a)
    conv3a = Conv2D(n_channel, (1, 1), strides=(1, 1), padding='same', activation='relu',
                    kernel_regularizer=regularizer, name=name + '_conv3a')(conv2a_bn)
    conv1b = Conv2D(n_channel, (1, 1), strides=strides, padding='same',
                    name=name + '_conv1b')(inlayer)
    res = Add(name=name + '_add')([conv3a, conv1b])
    return res


def res_net(features, regularizer=None):
    norm = Lambda(normalize, name='norm')(features)
    conv = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu',
                  kernel_regularizer=regularizer, name='conv')(norm)
    conv_bn = BatchNormalization(name='conv_bn')(conv)
    res1 = res_block(conv_bn, 16, strides=(1, 1), regularizer=regularizer, name='res1')
    res2 = res_block(res1, 32, strides=(2, 2), regularizer=regularizer, name='res2')
    res3 = res_block(res2, 32, strides=(1, 1), regularizer=regularizer, name='res3')
    res4 = res_block(res3, 64, strides=(2, 2), regularizer=regularizer, name='res4')
    res5 = res_block(res4, 64, strides=(1, 1), regularizer=regularizer, name='res5')
    pool = MaxPool2D((2, 1), name='pool')(res5)
    return pool


def gru_net(inlayer):
    '''
    inlayer is a res_net output;
    target_shape for GRU layers should be like (batch_size, seq_len, n_channel).
    '''
    # reshape should be like (batch_size, seq_len, n_channel).
    inlayer_shape = inlayer.get_shape()
    target_shape = (int(inlayer_shape[2]), int(inlayer_shape[1] * inlayer_shape[3]))
    reshaped = Reshape(target_shape, name='reshape')(inlayer)
    # gru layers
    gru1_f = GRU(128, return_sequences=True, name='gru1_f')(reshaped)
    gru1_b = GRU(128, return_sequences=True,
                 go_backwards=True, name='gru1_b')(reshaped)
    gru1 = Concatenate(name='gru1_add')([gru1_f, gru1_b])
    gru2_f = GRU(128, return_sequences=True, name='gru2_f')(gru1)
    gru2_b = GRU(128, return_sequences=True,
                 go_backwards=True, name='gru2_b')(gru1)
    gru2 = Concatenate(name='gru2_concat')([gru2_f, gru2_b])
    return gru2


class CodeDetector(object):

    def __init__(self, config):
        self.config = config
        self.input_shape = (32, 128, 3)
        self.n_class = len(CHARS) + 1
        self.regularizer = None
        self.build()

    def load_data(self):
        self.train_datagen = datagen(batch_size=self.config.train_batch_size)
        self.train_n_samples = int(12800 / self.config.train_batch_size)
        self.valid_datagen = datagen(batch_size=self.config.val_batch_size)
        self.valid_n_samples = 5

    def build(self):
        features = Input(shape=self.input_shape, name='features')
        res_out = res_net(features, self.regularizer)
        gru_out = gru_net(res_out)
        # softmax output
        predicts = Dense(self.n_class, activation='softmax', name='softmax_output')(gru_out)
        self.model = Model(features, predicts)

        labels = Input(name='labels', shape=[MAX_LEN], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss = Lambda(ctc_loss, output_shape=[1],
                      name='ctc')([predicts, labels, input_length, label_length])
        self.loss = Model(inputs=[features, labels, input_length, label_length], outputs=[loss])

    def show_model(self, to_png=True, show_loss=True):
        if show_loss:
            model = self.loss
        else:
            model = self.model
        for layer in model.layers:
            if hasattr(layer, 'trainable'):
                print('{:13s}\t{}'.format(layer.name, layer.trainable))
        if to_png:
            png_path = self.config.save_model_name + '.png'
            print('Painting model to {}'.format(png_path))
            plot_model(model, show_shapes=True, to_file=png_path)

    def train(self, fine_tune=False):
        '''Train or fine_tune the model.'''
        self.load_data()
        weights_path = self.config.save_model_name + '.h5'
        if fine_tune and os.path.exists(weights_path):
            print('Loading {}'.format(weights_path))
            self.loss.load_weights(weights_path)
            adam_lr = self.config.fine_tune_lr
        else:
            if fine_tune:
                print('No previous saved weights exist, train from scratch.')
            adam_lr = self.config.start_lr
        print('Using Adam optimizer with lr={:f}'.format(adam_lr))
        optimizer = Adam(lr=adam_lr)

        # compile model
        self.loss.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                          optimizer=optimizer)

        # callbacks
        save_best_model = ModelCheckpoint(weights_path,
                                          save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10)
        tensorboard_log = TensorBoard()
        callback_list = [save_best_model, reduce_lr,
                         tensorboard_log]

        # train ops
        self.loss.fit_generator(generator=self.train_datagen,
                                steps_per_epoch=self.train_n_samples,
                                epochs=self.config.epoches,
                                callbacks=callback_list,
                                validation_data=self.valid_datagen,
                                validation_steps=self.valid_n_samples)

    def load_model(self, weights_path=None):
        if weights_path is None:
            weights_path = self.config.save_model_name + '.h5'
        print('Loading {}'.format(weights_path))
        self.model.load_weights(weights_path)

    def predict(self, X):
        y_pred = self.model.predict(X)
        input_length = np.ones(y_pred.shape[0]) * y_pred.shape[1]
        predicts = K.eval(K.ctc_decode(y_pred, input_length)[0][0])
        return predicts

    def evaluate(self):
        raise NotImplementedError
