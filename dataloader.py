#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Sat Dec 16 2017 
@Time    : 18:22:47
@File    : dataloader.py
@Author  : alpha
"""

import numpy as np
import pandas as pd
import string
import random
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from captcha import codeGenerator

if os.sys.platform == 'win32':
    DEFAULT_FONTPATH = r'C:\Windows\Fonts\UbuntuMono-R.ttf'
else:
    DEFAULT_FONTPATH = r'/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf'

CHARS = string.digits + string.ascii_letters
MAX_LEN = 7


class CodeCooker(object):

    def __init__(self, fix_code=False, max_width=MAX_LEN, img_w=128, img_h=32):
        self.fix_code = fix_code
        self.max_width = max_width
        self.img_w = img_w
        self.img_h = img_h
        self.fontpaths = [DEFAULT_FONTPATH]

    def paint(self, auto=True, code='test', mode='RGB', color='rand'):
        if auto:
            self._gen_code()
        else:
            self.code = code
        if color == 'rand':
            color = self._randRGBcolor()
        else:
            color = color
        img = Image.new(mode=mode, size=(self.img_w, self.img_h),
                        color=color)
        draw = ImageDraw.Draw(img)
        fontpath = random.sample(self.fontpaths, 1)[0]
        fontsize = min(self.img_w, self.img_h) - 5
        font = ImageFont.truetype(fontpath, fontsize)
        _, height = font.getsize('hqikbfjplydg')
        width, _ = font.getsize(self.code)
        posw = (self.img_w - width) / 2
        posh = (self.img_h - height) / 2
        draw.text((posw, posh), self.code, font=font)
        img = np.array(img)
        return img

    def captcha(self, auto=True, code='test'):
        if auto:
            self._gen_code()
        else:
            self.code = code
        img = codeGenerator(text=self.code, bgsize=(self.img_w, self.img_h))
        return np.array(img)

    def _gen_code(self):
        source = list(CHARS)
        if not self.fix_code:
            self.code_width = random.randint(2, self.max_width)
        else:
            self.code_width = self.max_width
        self.code = ''.join([random.choice(source) for _ in range(self.code_width)])
        if random.randint(0, 1):
            code_list = list(self.code)
            pos = random.randint(0, len(code_list) - 2)
            code_list[pos + 1] = code_list[pos]
            self.code = ''.join(code_list)

    def _randRGBcolor(self):
        R = random.randrange(256)
        noR = 255 - R
        if noR == 0:
            G = 0
            B = 0
        else:
            G = random.randrange(noR)
            B = 255 - G
        return (R, G, B)

    def add_fonts(self, fontpath):
        self.fontpaths.append(fontpath)


def encode(codestr):
    return [CHARS.find(char) for char in codestr]


def decode(idxlist):
    idxlist = idxlist[idxlist != -1]
    return ''.join([CHARS[idx] for idx in idxlist])


def cal_label_length(batch_y):
    return np.array([len(y[y < len(CHARS)]) for y in batch_y])


def datagen(batch_size=128, max_len=MAX_LEN):
    cc = CodeCooker()
    while True:
        data = np.array([[cc.captcha(), cc.code] for i in range(batch_size)]).T
        batch_X = np.stack(data[0])
        input_length = np.ones(batch_size) * 32
        label_length = np.array([len(code) for code in data[1]], dtype='int64')
        batch_y = np.array([encode(code)+[-1]*(max_len-len(code)) \
                            for code in data[1]], dtype='int64')
        inputs = {'features': batch_X,
                  'labels': batch_y,
                  'input_length': input_length,
                  'label_length': label_length}
        outputs = {'ctc': np.zeros(batch_size)}  # dummy data for dummy loss function
        yield (inputs, outputs)


if __name__ == '__main__':
    for batch in datagen(32):
        break

    print(batch[1])
    # plt.subplot(2, 1, 1)
    # plt.imshow(X_test[0])
    # plt.title(decode(y_test[0]) + ' ' + str(y_test[0]))
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 1, 2)
    # plt.imshow(X_test[1])
    # plt.title(decode(y_test[1]) + ' ' + str(y_test[1]))
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    # print(y_test.shape)
    # print(label_length)
    # print([decode(y) for y in y_test])
    # print(cal_label_length(y_test))