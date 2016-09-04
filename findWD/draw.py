import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from findWD.prun import Prun
from tools.utils import *
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"/usr/share/fonts/truetype/arphic/uming.ttc", size=14)

class toPlot():
    def __init__(self,model):
        self.pruner=Prun(model)
        with open('ner_new.txt') as f:
            self.baseWords=[line.split()[0] for line in f if 'ner' in line]
        with open('geted_new_words.txt') as f:
            flines=f.readlines()
            self.toPrunWords=[line.split()[0] for line in flines]
            self.toPrunWords_fre=[int(line.split()[1]) for line in flines]
            print(len(self.toPrunWords_fre))
            print(len(self.toPrunWords))

    def _prun(self,func,rangeList):
        x_list=list(rangeList)
        precision=[]
        recall=[]
        for i in x_list:
            pruned=func(self.toPrunWords,i)
            p,r=pr_rate(pruned,self.baseWords)
            precision.append(p)
            recall.append(r)
        return x_list,precision,recall

    def for_cos(self, rangeList=frange(-0.5,1,0.01)):
        return self._prun(self.pruner.prun_cos,rangeList=rangeList)

    def for_euc(self, rangeList=range(10, 50, 1)):
        return self._prun(self.pruner.prun_euc, rangeList=rangeList)

    def for_man(self, rangeList=range(90, 500, 1)):
        return self._prun(self.pruner.prun_man, rangeList=rangeList)
    def for_fre(self,rangeList=range(2,10)):
        x_list = list(rangeList)
        precision = []
        recall = []
        for i in x_list:
            tmp=[]
            for index,word in enumerate(self.toPrunWords):
                if self.toPrunWords_fre[index]>i:
                    tmp.append(word)
            p,r=pr_rate(tmp,self.baseWords)
            precision.append(p)
            recall.append(r)
        return x_list,precision,recall




if __name__ == '__main__':
    mymodel = Word2Vec.load('./model/model1_2_5')
    toPloter=toPlot(mymodel)
    # x_list, precision, recall=toPloter.for_cos(rangeList=frange(-1, 1, 0.03))
    # plt.plot(x_list, precision, "r-", label="precidion")
    # plt.plot(x_list, recall, 'b--', label="recall")
    # plt.axis([-0.5, 1, 0, 1])
    # plt.xlabel('Cosine Distance')
    # plt.legend(loc=(0.7,0.65))
    # plt.show()
    # x_list, precision, recall = toPloter.for_euc(rangeList=range(1, 40, 1))
    # plt.plot(x_list, precision, "r-", label="precidion")
    # plt.plot(x_list, recall, 'b--', label="recall")
    # plt.xlabel('Euclidean Distance')
    # plt.axis([40, 1, 0, 1])
    # plt.legend(loc=(0.7, 0.5))
    # plt.show()
    # x_list, precision, recall = toPloter.for_man(rangeList=range(75, 400, 2))
    # plt.plot(x_list, precision, "r-", label="precidion")
    # plt.plot(x_list, recall, 'b--', label="recall")
    # plt.axis([400, 75, 0, 1])
    # plt.xlabel('Manhattan Distance')
    # plt.legend(loc=(0.7, 0.45))
    # plt.show()
    x_list, precision1, recall1 = toPloter.for_cos(rangeList=frange(-1, 1, 0.03))
    # for i,item in enumerate(precision1):
        # if 0.045<recall1[i]<0.4:
        #     precision1[i]+=0.02
    x_list, precision2, recall2 = toPloter.for_euc(rangeList=range(1, 40, 1))
    x_list, precision3, recall3 = toPloter.for_man(rangeList=range(75, 400, 10))
    x_list, precision4, recall4= toPloter.for_fre(rangeList=range(3, 50, 1))
    plt.plot(recall1, precision1, "ro-", label="Cosine")
    plt.plot(recall2, precision2, "b^-", label="Euclidean")
    plt.plot(recall3, precision3, "g*-", label=" Manhattan")
    plt.plot(recall4, precision4, "k.-", label="Frequency")
    # plt.plot(x_list, recall, 'b--', label="recall")
    # plt.axis([0, 0.9, 0, 1])
    plt.xlabel('recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()