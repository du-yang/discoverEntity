import numpy as np

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

def manDistance(vec1, vec2):
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec1, np.ndarray):
        vec2 = np.array(vec2)
    sqDiffVector = vec1 - vec2
    sqDiffVector = sqDiffVector ** 2
    distance=sqDiffVector ** 0.5
    distance = distance.sum()
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

