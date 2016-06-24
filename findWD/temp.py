from gensim.models import Word2Vec
from tools import *
import jieba.posseg as pseg

with open('taged_new_words.txt') as f:
    ner = [line.split()[0] for line in f if 'ner' in line]
print(len(set(ner)))

with open('geted_new_words.txt') as f:
    geted_words = [line.split() for line in f]

mymodel = Word2Vec.load('./model/model1_2_5')

words = [llist[0] for llist in geted_words]
frequency = [int(llist[1]) for llist in geted_words]
cos=[]
euc=[]
for item in words:
    item_split=item.split('##')
    vec1=mymodel[item_split[0]]
    vec2=mymodel[item_split[1]]
    cos.append(cosDistance(vec1,vec2))
for item in words:
    item_split=item.split('##')
    vec1=mymodel[item_split[0]]
    vec2=mymodel[item_split[1]]
    euc.append(eucDistance(vec1,vec2))
for i in range(len(words)):
    if euc[i]<14 and euc[i]>7:
        if words[i] not in ner:
            print(words[i],'----',frequency[i],cos[i],euc[i])
        if words[i] in ner:
            print(words[i], 'ner', frequency[i], cos[i], euc[i])
# words_flag=[]
# for word in words:
#     words_flag.append(str([word.word+'/'+word.flag for word in pseg.cut(word)]))
# for i in range(len(words)):
#     if words[i] in ner:
#         print(words_flag[i],'ner',cos[i],euc[i],frequency[i])
#     elif words[i] not in ner:
#         print(words_flag[i], '---', cos[i], euc[i], frequency[i])