import csv
from collections import OrderedDict

with open('word_pairs_strip.txt') as f:
    lines=[line.split() for line in f]

tmp_ditc={}
for ll in lines:
    tmp_ditc[tuple(ll[:2])]=int(ll[2])

pair_dict={}
pair_dict[('苹果','三星')]=0

for tkey in tmp_ditc:
    tlist=list(tkey)
    tlist[0],tlist[1]=tlist[1],tlist[0]
    tup1=tuple(tlist)
    if tkey not in pair_dict and tup1 not in pair_dict:
        pair_dict[tkey]=tmp_ditc[tkey]
    elif tkey in pair_dict:
        pair_dict[tkey]+=tmp_ditc[tkey]
    elif tup1 in pair_dict:
        pair_dict[tup1] += tmp_ditc[tkey]


pair_dict=OrderedDict(sorted(pair_dict.items(), key=lambda t: t[1],reverse=True))
# print(pair_dict[('雷军','小米')])
def center_one(entity,file):
    with open(file,'w') as w:
        writer=csv.writer(w)
        for key in pair_dict:
            l_tmp = []
            if entity==key[0] or entity==key[1]:
                if pair_dict[key]>5:
                    l_tmp.extend(list(key))
                    l_tmp.append(pair_dict[key])
                    if l_tmp[0]!=entity:
                        l_tmp[1],l_tmp[0]=l_tmp[0],l_tmp[1]
                    writer.writerow(l_tmp)
center_one('三星','sanxing.csv')
center_one('雷军','leijun.csv')
center_one('苹果','pinguo.csv')
center_one('小米','xiaomi.csv')
center_one('腾讯','tengxun.csv')
center_one('阿里巴巴','alibaba.csv')
center_one('阿里','ali.csv')

