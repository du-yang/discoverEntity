import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from findWD.prun import Prun
from tools.utils import *

class toPlot():
    def __init__(self,model):
        self.pruner=Prun(model)
        with open('taged_new_words.txt') as f:
            self.baseWords=[line.split()[0] for line in f if 'ner' in line]
            self.toPrunWords=[line.split()[0] for line in f]
            print(self.baseWords)
        # with open('geted_new_words.txt') as f:
        #     self.toPrunWords=[line.strip() for line in f]

    def plot_cos(self,rangeList=frange(-0.5,1,0.01)):
        x_list=list(rangeList)
        precision=[]
        recall=[]
        for i in x_list:
            pruned=self.pruner.prun_cos(self.toPrunWords,i)
            p,r=pr_rate(pruned,self.baseWords)
            precision.append(p)
            recall.append(r)
        return x_list,precision,recall

if __name__ == '__main__':
    mymodel = Word2Vec.load('./model/model1_2_5')
    toPloter=toPlot(mymodel)
    x_list, precision, recall=toPloter.plot_cos()
    plt.plot(x_list, precision, color="r", label="precidion")
    plt.plot(x_list, recall, 'b', label="recall")
    plt.axis([-0.5, 1.2, 0, 1])
    plt.xlabel('Cosine Distance')
    plt.legend()
    plt.show()