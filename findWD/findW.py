from tools import *
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt

def iterate_minibatches(inputs, batchsize, shuffle=False):
    '''
    批处理
    :param inputs:
    :param batchsize:
    :param shuffle:
    :return:
    '''
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


def get_array_data(file):
    with open(file) as f:
        data=[line.strip().split(' ') for line in f]
    return data


def main():
    mymodel = Word2Vec.load('./model/model1_2_5')
    freq = freqWord(6,0.0001,stopwordsFile='../config/stopWords.txt')
    data = get_array_data('../data/news_lines_splited.txt')[:5000]
    # for line in data:
    #     print(line)
    with open('geted_new_words.txt','w') as f:
        geted_words=[]
        for n,batch_data in enumerate(iterate_minibatches(data,5000)):
            new_words=freq.combine2words(batch_data)
            for item in new_words:
                tmp=[]
                if '##'.join(item) not in geted_words:
                    tmp.append('##'.join(item))
                    tmp.append(new_words[item][0])
                    geted_words.append(tmp)

                print(tmp)
                for index in range(len(item)-1):
                    print('cos:',cosDistance(mymodel[item[index]],mymodel[item[index+1]]))
                    print('euc:', eucDistance(mymodel[item[index]], mymodel[item[index + 1]]))
            print('完成第%s轮'%n)
        for words in geted_words:
            f.write(words[0]+' '+str(words[1])+'\n')
    with open('taged_new_words.txt') as f:
        total = [line.split()[0] for line in f if 'ner' in line]

    x=list(range(100))
    y1 = []
    y2 = []
    for i in range(100):
        ner_word_freq = []
        for item in geted_words:
            if item[1]>=i:
                ner_word_freq.append(item[0])
        precision, recall=pr_rate(ner_word_freq,total)
        y1.append(precision)
        y2.append(recall)
        print(i,precision, recall)
    plt.plot(x,y1,'b')
    plt.plot(x,y2,'r')
    plt.show()

    x = list(frange(-0.5,1,0.01))
    y1 = []
    y2 = []
    for i in frange(-0.5,1,0.01):
        ner_word_cos = []
        for item in geted_words:
            words=item[0].split('##')
            cos=cosDistance(mymodel[words[0]],mymodel[words[1]])
            if cos >= i:
                ner_word_cos.append(item[0])
        precision, recall = pr_rate(ner_word_cos, total)
        y1.append(precision)
        y2.append(recall)
        print(i, precision, recall)
    plt.plot(x, y1, 'b')
    plt.plot(x, y2, 'r')
    plt.show()

    x = list(range(0, 40))
    y1 = []
    y2 = []
    for i in range(0, 40):
        ner_word_euc = []
        for item in geted_words:
            words = item[0].split('##')
            euc = eucDistance(mymodel[words[0]], mymodel[words[1]])
            if euc <= i:
                ner_word_euc.append(item[0])
        precision, recall = pr_rate(ner_word_euc, total)
        y1.append(precision)
        y2.append(recall)
        print(i, precision, recall)
    plt.plot(x, y1, 'b')
    plt.plot(x, y2, 'r')
    plt.show()

    for i in frange(10, 50,1):
        ner_word_cos = []
        for item in geted_words:
            words = item[0].split('##')
            cos = cosDistance(mymodel[words[0]], mymodel[words[1]])
            euc = eucDistance(mymodel[words[0]], mymodel[words[1]])
            if 500*cos/euc+0.1*item[1] >= i:
                ner_word_cos.append(item[0])
        print(i, pr_rate(ner_word_cos, total))

    for i in frange(10, 100, 1):
        ner_word_cos = []
        for item in geted_words:
            words = item[0].split('##')
            cos = cosDistance(mymodel[words[0]], mymodel[words[1]])
            euc = eucDistance(mymodel[words[0]], mymodel[words[1]])
            if cos<0:cos=0.001
            if cos  >= 0.16 or item[1]>10+i:
                ner_word_cos.append(item[0])
        print(i, pr_rate(ner_word_cos, total))

    for i in frange(1, 3, 0.1):
        ner_word_cos = []
        for item in geted_words:
            words = item[0].split('##')
            cos = cosDistance(mymodel[words[0]], mymodel[words[1]])
            euc = eucDistance(mymodel[words[0]], mymodel[words[1]])
            if cos < 0: cos = 0.003
            if cos >= 0.16 or item[1] > (-np.log(cos)-i)*70 or euc<=12:
                ner_word_cos.append(item[0])
        print(i, pr_rate(ner_word_cos, total))
    for i in frange(1, 30, 1):
        ner_word_cos = []
        for item in geted_words:
            words = item[0].split('##')
            cos = cosDistance(mymodel[words[0]], mymodel[words[1]])
            euc = eucDistance(mymodel[words[0]], mymodel[words[1]])
            if cos < 0: cos = 0.003
            if cos >= 0.16 or euc <= i:
                ner_word_cos.append(item[0])
        print(i, pr_rate(ner_word_cos, total))


if __name__ == '__main__':
    main()