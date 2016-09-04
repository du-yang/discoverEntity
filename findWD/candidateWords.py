from tools import *
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt

def get_array_data(file):
    with open(file) as f:
        data=[line.strip().split(' ') for line in f]
    return data

def main():
    mymodel = Word2Vec.load('./model/model1_2_5')
    freq = freqWord(4,0.01,stopwordsFile='../config/stopWords1.txt')
    data = get_array_data('../data/news_lines_splited.txt')[:5000]
    num=0
    for line in data:
        num+=len(line)
    print(num)

'''
    with open('geted_new_words_fmi.txt','w') as f:
        geted_words=[]
        new_words=freq.combine2words(data)
        for item in new_words:
            tmp=[]
            if '##'.join(item) not in geted_words:
                tmp.append('##'.join(item))
                tmp.append(new_words[item][0])
                geted_words.append(tmp)

                f.write(tmp[0]+' '+str(tmp[1]))

            print(tmp)
            for index in range(len(item)-1):
                try:
                    print('cos:',cosDistance(mymodel[item[index]],mymodel[item[index+1]]))
                    print('euc:', eucDistance(mymodel[item[index]], mymodel[item[index + 1]]))
                    f.write(' '+str(cosDistance(mymodel[item[index]],mymodel[item[index+1]])))
                except:
                    pass
            f.write('\n')
'''
    # with open('taged_new_words.txt') as f:
    #     total = [line.split()[0] for line in f if 'ner' in line]
if __name__ == '__main__':
    main()