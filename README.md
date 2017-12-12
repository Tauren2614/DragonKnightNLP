# 简介
DragonKnightNLP是一个用python写的，专用于中文的处理库，参考了[SnowNLP](https://github.com/isnowfy/snownlp)的算法实现。


# 分词演示
```python
    nlp = DragonKnightNLP(u'工信处女干事每月经过下属科室都要亲口交代二十四口交换机等技术性器件的安装工作')
    for str in nlp.words():
        print(str)
```
结果：
```
工信处
女干事
每月
经过
下属
科室
都要
亲口
交代
二十四
口
交换机
等
技术性
器件
的
安装
工作
```