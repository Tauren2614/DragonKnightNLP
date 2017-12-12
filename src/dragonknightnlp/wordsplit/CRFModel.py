#coding=utf-8

from __future__ import unicode_literals
import sys
import gzip
import marshal
import os
import codecs
import re
from math import log

class BaseProb(object):
    def __init__(self):
        self.d = {}
        self.total = 0.0
        self.none = 0
    
    def exists(self,key):
        return key in self.d
    
    def getsum(self):
        return self.total
    
    def get(self,key):
        if not self.exists(key):
            return False,self.none
        return True,self.d[key]
    
    def freq(self,key):
        return float(self.get(key)[1])/self.total
    
    def samples(self):
        return self.d.keys()

class NormalProb(BaseProb):
    def add(self,key,value):
        if not self.exists(key):
            self.d[key] = 0
        self.d[key] = value
        self.total += value

class CharacterBasedGenerativeModel(object):
    def __init__(self):
        self.l1 = 0.0
        self.l2 = 0.0
        self.l3 = 0.0
        self.status = ('b','m','e','s')
        #单个字统计
        self.uni = NormalProb()
        #单字与前一个字
        self.bi = NormalProb()
        #单字与前两个字
        self.tri = NormalProb()

    def save(self, fname, iszip=True):
        d = {}
        for k, v in self.__dict__.items():
            if hasattr(v, '__dict__'):
                d[k] = v.__dict__
            else:
                d[k] = v
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            marshal.dump(d, open(fname, 'wb'))
        else:
            f = gzip.open(fname, 'wb')
            f.write(marshal.dumps(d))
            f.close()

    def load(self, fname, iszip=True):
        if sys.version_info[0] == 3:
            fname = fname+'.3'
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            try:
                f = gzip.open(fname, 'rb')
                d = marshal.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = marshal.loads(f.read())
            f.close()
        for k, v in d.items():
            if hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v

    def log_prob(self, s1, s2, s3):
        #单个字概率
        uni = self.l1*self.uni.freq(s3)
        #s2s3连起来相对于s2单个字的概率
        bi = self.div(self.l2*self.bi.get((s2, s3))[1], self.uni.get(s2)[1])
        #s1s2s3连起来相对于s1s2的概率
        tri = self.div(self.l3*self.tri.get((s1, s2, s3))[1],
                       self.bi.get((s1, s2))[1])
        #forwoard ?????? 没搞明白
        if uni+bi+tri == 0:
            return float('-inf')
        return log(uni+bi+tri)

    def tag(self, data):
        """
        to: 给传入字串的每一个字打上标签
        """
        now = [((('', 'BOS'), ('', 'BOS')), 0.0, [])]
        for w in data:
            stage = {}
            not_found = True
            for s in self.status:
                if self.uni.freq((w, s)) != 0:
                    not_found = False
                    break
            if not_found:
                for s in self.status:
                    for pre in now:
                        stage[(pre[0][1], (w, s))] = (pre[1], pre[2]+[s])
                now = list(map(lambda x: (x[0], x[1][0], x[1][1]),
                               stage.items()))
                continue
            for s in self.status:
                for pre in now:
                    p = pre[1]+self.log_prob(pre[0][0], pre[0][1], (w, s))
                    if (not (pre[0][1],
                             (w, s)) in stage) or p > stage[(pre[0][1],
                                                            (w, s))][0]:
                        stage[(pre[0][1], (w, s))] = (p, pre[2]+[s])
            now = list(map(lambda x: (x[0], x[1][0], x[1][1]), stage.items()))
        return zip(data, max(now, key=lambda x: x[1])[2])

    def div(self, v1, v2):
        if v2 == 0:
            return 0
        return float(v1)/v2

    def train(self, fname):
        #加载训练文件
        fr = codecs.open(fname, 'r', 'utf-8')
        data = []
        for i in fr:
            line = i.strip()
            if not line:
                continue
            tmp = map(lambda x: x.split('/'), line.split())
            data.append(tmp)
        fr.close()
        
        for sentence in data:
            now = [('', 'BOS'), ('', 'BOS')]
            self.bi.add((('', 'BOS'), ('', 'BOS')), 1)
            self.uni.add(('', 'BOS'), 2)
            for word, tag in sentence:
                now.append((word, tag))
                self.uni.add((word, tag), 1)
                self.bi.add(tuple(now[1:]), 1)
                self.tri.add(tuple(now), 1)
                now.pop(0)
        tl1 = 0.0
        tl2 = 0.0
        tl3 = 0.0
        #按照三连词出现的频率排序
        samples = sorted(self.tri.samples(), key=lambda x: self.tri.get(x)[1])
        for now in samples:
            #三连词相对于二连词的比率
            c3 = self.div(self.tri.get(now)[1]-1, self.bi.get(now[:2])[1]-1)
            #二连词相对于单个字的比率
            c2 = self.div(self.bi.get(now[1:])[1]-1, self.uni.get(now[1])[1]-1)
            #单个字相对于所有字的比率
            c1 = self.div(self.uni.get(now[2])[1]-1, self.uni.getsum()-1)
            #三连字出现几率大于二连词大于单个字出现次数，说明其是三连词
            if c3 >= c1 and c3 >= c2:
                tl3 += self.tri.get(now)[1]
            elif c2 >= c1 and c2 >= c3:
                tl2 += self.tri.get(now)[1]
            elif c1 >= c2 and c1 >= c3:
                tl1 += self.tri.get(now)[1]
        #单个字所占比率
        self.l1 = self.div(tl1, tl1+tl2+tl3)
        #两字词比率
        self.l2 = self.div(tl2, tl1+tl2+tl3)
        #三连字比率
        self.l3 = self.div(tl3, tl1+tl2+tl3)

    def cut(self, sent):
        """
        to: 切割中文字串
        """
        words = []
        #把文字按照非中文字符分割，确保处理的是连续的中文
        re_zh = re.compile(u'([\u4E00-\u9FA5]+)')
        for s in re_zh.split(sent):
            s = s.strip()
            if not s:
                continue
            if re_zh.match(s):
                #分割处理连续的中文
                words += self.single_seg(s)
            else:
                #处理非中文，非中文直接作为一个分词
                for word in s.split():
                    word = word.strip()
                    if word:
                        words.append(word)
        return words

    def single_seg(self, sent):
        return list(self.seg(sent))

    def seg(self, sentence):
        #逐字打上标签
        ret = self.tag(sentence)
        tmp = ''
        #按照标签将字串切成一个个词
        for i in ret:
            if i[1] == 'e':
                yield tmp+i[0]
                tmp = ''
            elif i[1] == 'b' or i[1] == 's':
                if tmp:
                    yield tmp
                tmp = i[0]
            else:
                tmp += i[0]
        if tmp:
            yield tmp


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'seg_v2.marshal')
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data.txt')
    mode = CharacterBasedGenerativeModel()
    mode.train(train_path)
    mode.save(data_path)
    mode.load(data_path)
    words = mode.cut(u'工信处女干事每月经过下属科室都要亲口交代二十四口交换机等技术性器件的安装工作')
    for sr in words:
        print (sr)
    #for key in mode.uni.d:
    #    print ' '+key[0]+' '+key[1]+":"+str(mode.uni.get(key)[1])

