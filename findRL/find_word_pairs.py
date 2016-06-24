import jieba.analyse
import jieba
import numpy as np

from collections import Counter

jieba.load_userdict('word_dic.txt')

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

def filter_count(pair_dic,n):
    filtered={}
    for key in pair_dic:
        if pair_dic[key]>n:
            filtered[key]=pair_dic[key]
    return sorted(filtered.items(),key=lambda d:d[1],reverse=True)

def find_pair(lines_file):
    with open(lines_file) as f:
        lines=[line.strip() for line in f]

    lines_batchs=iterate_minibatches(lines,5000)
    pairs_dic=Counter()

    for ii,lines_batch in enumerate(lines_batchs):
        pair_dic=Counter()
        for line in lines_batch:
            word_pairs=set()
            line = jieba.analyse.extract_tags(line,allowPOS=('ner','nrt','n','ns','nt','nz'))

            for index,word in enumerate(line):
                for i in range(index+1,len(line)):
                    tmp=[]
                    tmp.append(word)
                    tmp.append(line[i])
                    word_pairs.add(tuple(tmp))

            pair_dic += Counter(word_pairs)

        print('扫描完第%s批' % str(ii+1))
        pairs_dic += pair_dic
        # tmp=map(lambda x:dict(zip([x[0]],[x[1]])) ,filter(lambda d:d[1]>1,pairs_dic.items()))
        # tmp_coun=Counter()
        # for dic in tmp:
        #     tmp_coun+=Counter(dic)
        # pairs_dic=tmp_coun
        print(pairs_dic.most_common(n=5))
    return pairs_dic
pairs_dic = find_pair('text_for_wordvec.txt')
to_write=filter_count(pairs_dic,15)
with open('word_pairs.txt','w') as w:
    for pair in to_write:
        w.write(str(pair[0])+' '+str(pair[1])+'\n')