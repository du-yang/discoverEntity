# import jieba
#
# jieba.load_userdict('word_dic.txt')
import re

def tmp():
    with open('text_for_wordvec.txt','w') as f:
        with open('../findWD/news_lines.txt') as fr:
            lines=[line.strip() for line in fr]
        for line in lines:
            line = ' '.join(jieba.cut(line))
            f.write(line+'\n')

if __name__ == '__main__':
    with open('word_pairs.txt') as r:
        lines=[line.strip().split() for line in r]
        lines=[[word.strip().strip('(').strip(')').strip(',').strip("'") for word in line] for line in lines]

    with open('special_words.txt') as r:
        special_words=[word.strip() for word in r]

    with open('word_pairs_strip.txt','w') as w:
        for line in lines:
            if line[0] not in special_words and line[1] not in special_words:
                w.write(' '.join(line)+'\n')


