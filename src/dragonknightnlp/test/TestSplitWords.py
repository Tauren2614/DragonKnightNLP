# -*- coding: utf-8 -*-
from __future__ import unicode_literals
'''
Created on 2017-12-12

@author: Tauren
'''
from dragonknightnlp import DragonKnightNLP

if __name__ == '__main__':
    nlp = DragonKnightNLP(u'工信处女干事每月经过下属科室都要亲口交代二十四口交换机等技术性器件的安装工作')
    for str in nlp.words():
        print(str)