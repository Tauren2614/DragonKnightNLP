# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from . import CRFModel
import os
'''
分词
'''
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'seg_v2.marshal')
train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data.txt')
mModel = CRFModel.CharacterBasedGenerativeModel()
mModel.load(data_path)


def cut(sent):
    return mModel.cut(sent)

def train():
    mModel.train(train_path)
    mModel.save(data_path)
    mModel.load(data_path)