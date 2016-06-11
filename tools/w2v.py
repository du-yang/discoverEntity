import numpy as np
import gensim
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

def w2v(data,repeat=2,window=5):
    '''
    训练word2vec模型，供后面使用
    :param data:要求数据是已经分好的词，array-like的数据格式
    :param repeat: 在同一个数据上训练多少词
    :return: 返回训练好的模型
    '''

    dataList=np.array(data)
    w2v_model = gensim.models.Word2Vec(size=200,min_count=3,window=5)
    w2v_model.build_vocab(dataList)
    for epoch in range(repeat):
        perm = np.random.permutation(dataList.shape[0])
        w2v_model.train(dataList[perm])
    return w2v_model
