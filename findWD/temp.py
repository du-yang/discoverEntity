# with open('taged_new_words.txt') as f:
#     ner=0
#     no=0
#     lines=[line.strip() for line in f]
#     for line in lines:
#         if 'ner' in line:
#             ner+=1
#             print(line)
#         elif 'no' in line:
#             no+=1
#     print('ner:',ner)
#     print('no:',no)



# with open('taged_new_words.txt','w') as f:
#     for line in lines:
#         if 'ner' in line:
#             f.write(line+'\n')
#         else:
#             f.write(line+' '+'no'+'\n')
with open('taged_new_words.txt') as f:
    total = [line.split()[0] for line in f if 'ner' in line]
with open('geted_new_words.txt') as f:
    words = [word.strip() for word in f]
A=[word for word in words if word in total]
print(len(A))
print(len(words))
precision=len(A)/len(words)
recall=len(A)/len(total)
print('precision:',precision)
print('recall:',recall)
print(words)
print(A)

