from tools import w2v
from gensim.models import Word2Vec
import numpy as np

# model=w2v('text_for_wordvec.txt',2,window=5)
# model.save('model1_2_5')

def cosin_similariy(vec1, vec2):
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec1, np.ndarray):
        vec2 = np.array(vec2)
    return sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

mymodel = Word2Vec.load('./model/model1_2_5')
a=mymodel.most_similar(positive=['马云', '软银'], negative=['阿里巴巴'])

print('马云',mymodel.most_similar('马云'))
print('张小龙',mymodel.most_similar('张小龙'))
print('丁磊',mymodel.most_similar(['丁磊']))
print('雷军',mymodel.most_similar('雷军'))
print('马化腾',mymodel.most_similar('马化腾'))
print('申通',mymodel.most_similar('申通'))
print('圆通',mymodel.most_similar('圆通'))
print('顺丰',mymodel.most_similar('顺丰'))
print('----------------------------------------')
print(a)



























