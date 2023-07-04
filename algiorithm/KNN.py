import numpy as np
from sklearn import datasets
from sklearn import neighbors

class KNN():
    def __init__(self):
        #默认使用iris数据集
        self.iris = datasets.load_iris()
        self.X = self.iris.data
        self.y = self.iris.target

    def run(self):
        kn_clf = neighbors.KNeighborsClassifier()
        kn_clf.fit(self.X[:-20], self.y[:-20])
        y_pred = kn_clf.predict(self.X[-20:])
        return self.y[-20:] , y_pred
