import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from tools import *

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
assert len(words)==len(frequency)==len(cos)==len(euc)

def plot_line(words,base_words,rangeList,measure,control=1):
    for_list=list(rangeList)
    precision = []
    recall = []
    for point in for_list:
        if control==1:
            ner_words = [words[i] for i in range(len(words)) if measure[i] > point]
        elif control == 0:
            ner_words = [words[i] for i in range(len(words)) if measure[i] < point]
        precision1, recall1 = pr_rate(ner_words, base_words)
        precision.append(precision1)
        recall.append(recall1)
    plt.plot(for_list,precision,color="r",label="precidion")
    plt.plot(for_list,recall, 'b',label="recall")

    return precision,recall
    # plt.text(for_list[-2],precision[-2],r'presision')
    # plt.legend()
    # plt.show()



# for i in range(len(words)):
#     if words[i] in ner:
#         plt.plot(cos[i],frequency[i],'ro')
#     else:
#         plt.plot(cos[i],frequency[i], 'b^')
#     if  frequency[i]>20 and words[i] not in ner:
#         print(words[i],cos[i],euc[i],frequency[i],'no')
#
# y1 = []
# y2 = []
cos_p,cos_r=plot_line(words,ner,frange(-0.5,1,0.01),cos)
plt.axis([-0.5,1.2,0,1])
plt.xlabel('Cosine Distance')
plt.legend()
plt.show()

fre_p,fre_r=plot_line(words,ner,frange(0,100,1),frequency)
plt.xlabel('Frequency')
plt.legend()
plt.show()

euc_p,euc_r=plot_line(words,ner,range(1,50,1),euc,control=0)
plt.xlabel('Euclidean Distance')
plt.axis([50,-15,0,1])
plt.legend()
plt.show()

print(len(cos_r),len(fre_r),len(euc_r))
# plt.plot(cos_r,cos_p,'r',fre_r,fre_p,'b',euc_r,euc_p,'g',label='cos')
plt.plot(cos_r,cos_p,'r',label='Cosine')
plt.plot(fre_r,fre_p,'b',label='Frequency')
plt.plot(euc_r,euc_p,'g',label='Euclidean')
plt.xlabel('recall')
plt.ylabel('precidion')
plt.legend()
plt.show()




