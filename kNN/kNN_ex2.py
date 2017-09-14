'''
对应练习2.2 约会网站预测函数
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator

from pip._vendor.distlib.compat import raw_input

import kNN as k


# 将文本文件记录转换为标准数据格式
def file2matrix(filename):
    f = open(filename)
    lines = f.readlines()
    numOfLines = len(lines)  # 获得数据的行数
    returnMat = zeros((numOfLines, 3))
    classLabelVector = []
    index = 0
    for line in lines:
        line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 将数据集进行归一化处理：newValue = (oldValue-min)/(max-min)，使得数据集中在-1~1之间，所有特征的权重相等
def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    normDataSet = zeros(shape(dataSet))  # 初始化一个全为零的和dataSet一样大小的数组
    m = dataSet.shape[0]  # 获得行数
    # 这里没有用for循环，而是直接用矩阵运算
    normDataSet = dataSet - tile(minVal, (m, 1))  # 获得差
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 进行除法运算
    return normDataSet, ranges, minVal


# 测试数据
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errCount = 0.0
    for i in range(numTestVecs):
        classifierResult = k.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errCount += 1.0
    print("the total error rate is: %f" % (errCount / float(numTestVecs)))


# 运用到实际问题：约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']

    #进行输入
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))

    #获得归一化的数据集
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    inArr = array([ffMiles, percentTats, iceCream, ])
    #需要将inArr进行归一化处理
    classifierResult = k.classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: %s" % resultList[classifierResult - 1])

# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()

# datingClassTest()

classifyPerson()
