# -*- coding: utf-8 -*-

from itertools import chain
import jieba
import jieba.posseg as pseg
import re

class transData():
    '''
    把数据转换成想要的格式
    '''
    def __init__(self):
        pass

    @classmethod
    def doc2list(cls,file):
        '''数据格式为每个doc为一行'''
        with open(file,encoding='utf-8') as f:
            texts = [re.split(u'。|？|！|\n', doc.strip()) for doc in f if doc.strip()]
            return chain.from_iterable(texts)


    @classmethod
    def cut_word(cls,file,user_dict=False,seg=False):
        '''
        用结巴jieba分词进行分词
        '''
        if  not user_dict:
            pass
        else: jieba.load_userdict(user_dict)#添加用户词典
        if seg:
            return [[word.word+'/'+word.flag for word in pseg.cut(line,HMM=False)] for line in cls.doc2list(file) if line.strip()]
        else:
            return [[word for word in jieba.cut(line,HMM=False)] for line in cls.doc2list(file) if line.strip()]



