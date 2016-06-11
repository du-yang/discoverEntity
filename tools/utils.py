import numpy as np

def cosDistance(vec1, vec2):
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec1, np.ndarray):
        vec2 = np.array(vec2)
    return sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def eucDistance(vec1, vec2):
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec1, np.ndarray):
        vec2 = np.array(vec2)
    sqDiffVector = vec1-vec2
    sqDiffVector=sqDiffVector**2
    sqDistances = sqDiffVector.sum()
    distance = sqDistances**0.5
    return distance

def pr_rate(listA,listBase):
    '''
    :param listA:待计算准确率和召回率的列表
    :param listB: 计算准确率和召回率的基准列表，即所有的正确的集合
    :return: 准确率和召回率
    '''
    correct=[word for word in listA if word in listBase]
    precision=len(correct)/len(listA)
    recall = len(correct)/len(listBase)
    return precision,recall

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

