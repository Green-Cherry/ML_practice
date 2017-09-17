#coding=utf-8
'''
4.6 使用朴素贝叶斯过滤垃圾邮件
(1)收集数据：提供文本文件。
(2)准备数据：将文本文件解析成词条向量。
(3)分析数据：检查词条确保解析的正确性。
(4)训练算法：使用我们之前建立的trainNB0()函数。
(5)测试算法：使用clasSifyNB()，并且构建一个新的测试函数来计算文档集的错误率。
(6)使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上。
'''
from bayes import bayes_ex as bayes
from numpy import *

def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)  # create vocabulary
    trainingSet = range(50);
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print( "classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText

# spamTest();