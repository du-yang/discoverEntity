from tools.utils import *
from gensim.models import Word2Vec

class Prun():
    def __init__(self,model):
        self.model=model

    def prun_cos(self,toPrunList,threshold):
        prunedList=[]
        for word in toPrunList:
            word_l=word.split('##')
            tmp=[]
            flag=0
            for index in range(len(word_l)-1):
                if cosDistance(self.model[word_l[index]],self.model[word_l[index+1]])<=threshold:
                    tmp.append(word_l[flag:index+1])
                    flag=index+1
            tmp.append(word_l[flag:index+2])
            for word in tmp:
                if len(word)>1:
                    prunedList.append('##'.join(word))
        return prunedList

    def prun_euc(self,toPrunList, threshold):
        prunedList = []
        for word in toPrunList:
            word_l = word.split('##')
            tmp = []
            flag = 0
            for index in range(len(word_l) - 1):
                if eucDistance(self.model[word_l[index]], self.model[word_l[index + 1]]) >= threshold:
                    tmp.append(word_l[flag:index + 1])
                    flag = index + 1
            tmp.append(word_l[flag:index + 2])
            for word in tmp:
                if len(word) > 1:
                    prunedList.append('##'.join(word))
        return prunedList

    def prun_man(self, toPrunList, threshold):
        prunedList = []
        for word in toPrunList:
            word_l = word.split('##')
            tmp = []
            flag = 0
            for index in range(len(word_l) - 1):
                if manDistance(self.model[word_l[index]], self.model[word_l[index + 1]]) >= threshold:
                    tmp.append(word_l[flag:index + 1])
                    flag = index + 1
            tmp.append(word_l[flag:index + 2])
            for word in tmp:
                if len(word) > 1:
                    prunedList.append('##'.join(word))
        return prunedList


if __name__ == '__main__':
    with open('nerWords.txt') as f:
        ner = [line.split()[0] for line in f if 'ner' in line]

    mymodel = Word2Vec.load('./model/model1_2_5')
    pruned=Prun(mymodel)
    for word in pruned.prun_euc(ner,10):
        print(word)

                

