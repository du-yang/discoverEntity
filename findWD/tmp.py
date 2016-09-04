from tools.utils import *
with open('ner_new.txt') as f:
    baseWords = [line.split()[0] for line in f if 'ner' in line]
with open('geted_new_words_fmi.txt') as f:
    getWords = [line.split()[0] for line in f]
print(pr_rate(getWords,baseWords))
