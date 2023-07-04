import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

class SVM():
 def __init__(self):
  # 默认使用iris数据集
  self.iris = datasets.load_iris()
  self.X = self.iris.data[:, :2]
  self.y = self.iris.target
 def run(self):
  svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(self.X, self.y)
  # create a mesh to plot in
  x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
  y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
  h = (x_max / x_min) / 100
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  # plt.subplot(1, 1, 1)
  Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  return xx,yy,Z ,self.X ,self.y

if __name__ == '__main__':
 nb = SVM()
 xx,yy,Z ,X ,y = nb.run()

 plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
 plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
 # plt.xlabel('Sepal length')
 # plt.ylabel('Sepal width')
 # # 限制x的取值范围，便于显示
 # plt.xlim(xx.min(), xx.max())
 plt.title('LinearSVC test result')
 plt.show()

 # print(np.arange(y.shape[0]))
 # print(y_pred)


