from tools import w2v
from gensim.models import Word2Vec

with open('../data/news_lines_splited.txt', encoding='utf8') as f:
    data= [line.strip().split() for line in f]

model=w2v(data,repeat=2,window=5)
model.save('./model/model1_2_5')

mymodel = Word2Vec.load('./model/model1_2_5')