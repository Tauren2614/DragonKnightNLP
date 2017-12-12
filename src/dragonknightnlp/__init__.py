# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from . import wordsplit

class DragonKnightNLP(object):
    def __init__(self, doc):
        self.doc = doc
    
    def words(self):
        return wordsplit.cut(self.doc)