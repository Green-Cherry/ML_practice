'''
练习3.1.1 生成决策树
'''
from math import log
import operator

# 创建简易数据集,最后一列是类别
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 根据公式计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]  # 这样拆成两步，主要是将featVec[axis]不算在数据集中，因为这个是作为给定的特征
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1       #减一是因为最后一列是类别，不是特征值
    baseEntropy = calcShannonEnt(dataSet)   #计算基本熵
    bestInfoGain = 0.0                      #信息增益的变量
    bestFeature = -1                        #记录最好的特征是哪个
    for i in range(numFeatures):
        featlist = [example[i] for example in dataSet]      #获取数据集中某个特征的所有值
        uniqueVals = set(featlist)          #获取数据集中这个特征的所有无序不重复的值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))    #计算当前特征值的具体值所占的比重
            newEntropy += prob * calcShannonEnt(subDataSet) #计算这类特征值的香农熵
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):                       #比较所有特征中的信息增益，返回最好特征划分的索引值
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#获取类标签出现频率最高的类标签
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#构造决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #类别相同，停止划分
        return classList[0]
    if len(dataSet[0]) == 1:                            #如果特征值遍历完还没有出现类别相同的，就选择次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)        #选择最好特征值的序号
    bestFeatLabel = labels[bestFeat]                    #获得最好的特征值
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:                            #根据当前特征值的每个具体值，去判断是否能够确定最终类别
        subLabels = labels[:]                           #不能直接就=labels，因为这样相当于拿到引用，会改变原来的labels里面的值
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

#使用决策树的分类函数，传入决策树，特征值，测试的特征值，输出类别
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

#将生成的决策树存储起来
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)      #将决策树进行序列化存储
    fw.close()

#取出决策树
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)