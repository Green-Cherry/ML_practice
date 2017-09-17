#coding=utf-8

from bayes import bayes_ex as bayes
from bayes import bayes_ex2 as bayes2
from numpy import *
import feedparser

#遍历词汇表中的每个词，并统计它在文本中出现的次数，然后根据出现次数从高到低对词典进行排序，最后返回排序最高的30个单词


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = bayes2.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = bayes2.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)  # create vocabulary

    #####                       此段移除高频词                         #####
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    #####                       此段移除高频词                         #####

    trainingSet = list(range(2 * minLen))
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNB0(array(trainMat), array(trainClasses))

    #计算错误率
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
print(len(ny['entries']))
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
getTopWords(ny,sf)
