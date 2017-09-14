from numpy import *
import operator


#生成数据集（很小）
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

#kNN算法
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]        #获取第一维度的长度，这里就是获取行数
    diffMat=tile(inX,(dataSetSize,1))-dataSet       #用于获得两个矩阵（目标矩阵和训练数集矩阵）相减的值
    sqDiffMat=diffMat**2        #矩阵乘积
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1    #计算前K个里面每种类别的数量
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #对类别进行排序，选择数量最多的类别
    return sortedClassCount[0][0]

(dataSet,labels)=createDataSet()
inX=[0,0]
result=classify0(inX,dataSet,labels,3)
print(result)