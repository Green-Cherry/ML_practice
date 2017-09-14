'''
对应练习2.1 手写数字识别
'''
from os import listdir

from numpy import *

# 将图像格式化为向量。将32*32的二进制图像矩阵转化成1*1024的向量
# 因为分类器是二维的，一行对应一个类别，所以需要将图像转化为1*1024的向量
import kNN


def img2vector(filename):
    returnVect = zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 手写数字识别，测试分类器
def handwritingClassTest():
    hwLabels = []
    traingFileList = listdir('digits/trainingDigits')
    m = len(traingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with :%d, the real answer is: %d' % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n the total number of errors is: %d" % errorCount)
    print("\n the total error rate is: %f" % (errorCount/float(mTest)))

handwritingClassTest()