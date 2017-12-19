#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Mon Dec 18 2017 
@Time    : 22:47:31
@File    : infer.py
@Author  : alpha
"""

from model import CodeDetector
from dataloader import datagen, decode

import matplotlib.pyplot as plt
import numpy as np

num = 12
for test_data, _ in datagen(num):
    break

test_X = test_data['features']
test_y = test_data['labels']


class Config:
    save_model_name = 'model'


myocr = CodeDetector(Config())
myocr.load_model()
predicts = myocr.predict(test_X)

plt.figure(figsize=(4 * np.sqrt(num), 2.4 * np.sqrt(num)))

for i in range(num):
    plt.subplot(np.ceil(np.sqrt(num)).astype('int'),
                np.round(np.sqrt(num)).astype('int'), i + 1)
    plt.imshow(test_X[i], interpolation='bilinear')
    plt.xticks([]), plt.yticks([])
    plt.title('Truth: {}'.format(decode(test_y[i])), fontsize=20)
    plt.xlabel('Predict: {}'.format(decode(predicts[i])), fontsize=20)
    plt.ylabel('{}'.format(decode(test_y[i]) == decode(predicts[i])), fontsize=20)

plt.tight_layout()
plt.savefig('result.png')
# plt.show()
