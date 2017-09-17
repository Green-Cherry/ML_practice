#coding=utf-8
'''

4.5.1 朴素贝叶斯分类器 判断侮辱性文档

(1)收集数据：可以使用任何方法。
(2)准备数据：需要数值型或者布尔型数据。
(3)分析数据：有大量特征时，绘制特征作用不大，此时使用直方图效果更好。
(4)训练算法：计算不同的独立特征的条件概率。
(5)测试算法：计算错误率。
(6)使用算法：一个常见的朴素贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯分类器，不一定非要是文本。
'''

from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


# 构建单词集
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets，获得并集，去除重复元素
    return list(vocabSet)


# 若单词出现就记为1，没有出现记为0
# 朴素贝叶斯词集模型
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print
        "the word: %s is not in my Vocabulary!" % word
    return returnVec


# 获得概率，其中传入的第一个参数是文档数字化的矩阵，01矩阵，第二个参数是对应每篇文章的类别
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 侮辱性文档的概率
    p0Num = ones(numWords);
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0;
    p1Denom = 2.0  # change to 2.0，因为如果初始化是0，容易导致错误
    for i in range(numTrainDocs):  # 下面的加法都是向量加法
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # change to log() 计算的是侮辱性文档中词出现的概率
    p0Vect = log(p0Num / p0Denom)  # change to log() 计算的是正常性文档中词出现的概率
    return p0Vect, p1Vect, pAbusive

#最后根据哪个概率高，决定分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#测试文档是否为侮辱性的
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))     #将文档转化为数字向量
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


