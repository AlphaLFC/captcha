#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Sat Dec 16 2017 
@Time    : 18:15:36
@File    : listutils.py
@Author  : alpha
"""


def charlist(in_list):
    """
    Convert a list of strings to a list of separate chars.
    # Example
        param in_list: ['', 'abc', 'd']
        return: ['', 'a', 'b', 'c', 'd']
    """
    out_list = []
    for i in in_list:
        assert type(i) in [str]
        if i == '':
            out_list.append(i)
        else:
            out_list += list(i)
    return out_list


def flatten(in_list):
    '''Flatten a list[tuple] of list[tuple] objects to
    a list of non-list or non-tuple elements.

    # Example
        in_list = [1, [2, [3, 4]]]
        out_list = flattern(in_list)
        out_list -> [1, 2, 3, 4]
    '''
    out_list = []
    for i in in_list:
        if type(i) in [list, tuple]:
            out_list += flatten(i)
        else:
            out_list.append(i)
    return out_list


def shrinkstrlist(in_list):
    '''Shrink a list of str sequence to
    a list of strings seperated by empty str in the previous sequence.

    # Example
        in_list = ['', 'a', 'b', 'c', '', 'd']
        out_list = shrinkstrlist(in_list)
        out_list -> ['abc', 'd']
    '''
    tmp_list = []
    for i in in_list:
        if i == '':
            tmp_list.append(' ')
        else:
            tmp_list.append(i)
    outstr = ''.join(tmp_list)
    out_list = outstr.strip().split()
    return out_list
