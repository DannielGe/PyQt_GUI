# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

class DT():
    def __init__(self):
        #默认使用iris数据集
        self.iris = load_iris()

    def run(self):
        self.clf = tree.DecisionTreeClassifier(random_state=0,max_depth=3)
        self.clf = self.clf.fit(self.iris.data[:-20], self.iris.target[:-20])
        y_pred=self.clf.predict(self.iris.data[-20:])
        return self.iris.target[-20:] , y_pred
# #----------------数据准备----------------------------
# iris = load_iris()                          # 加载数据
# #---------------模型训练---------------------------------
# clf = tree.DecisionTreeClassifier(random_state=0,max_depth=3)
# clf = clf.fit(iris.data, iris.target)
# #---------------树结构可视化-----------------------------
# dot_data = tree.export_graphviz(clf)
# graph = graphviz.Source(dot_data)

