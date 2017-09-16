'''
练习3.4 隐形眼镜数据集
'''

from decision_tree import creatTreePicture
from decision_tree import tree

fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=tree.createTree(lenses,lensesLabels)
creatTreePicture.createPlot(lensesTree)