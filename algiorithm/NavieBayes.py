import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

class NB():
    def __init__(self):
        #默认使用iris数据集
        self.iris = datasets.load_iris()

    def run(self):
        clf = GaussianNB()
        clf = clf.fit(self.iris.data[:-20], self.iris.target[:-20])
        y_pred=clf.predict(self.iris.data[-20:])
        return self.iris.target[-20:] , y_pred


if __name__ == '__main__':
    nb = NB()
    y , y_pred = nb.run()
    # print(np.arange(y.shape[0]))
    print(y_pred)
