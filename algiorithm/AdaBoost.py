
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# 创建随机数种子

class ADB():
	def __init__(self):
		rng = np.random.RandomState(111)
		# 训练集X为300个0到10之间的随机数
		self.X = np.linspace(0, 10, 300)[:, np.newaxis]
		# 定义训练集X的目标变量
		self.y = np.sin(1 * self.X).ravel() + np.sin(2 * self.X).ravel() + np.sin(3 * self.X).ravel() + np.cos(3 * self.X).ravel() \
		    + rng.normal(0,0.3,self.X.shape[0])

	def run(self):
		adb = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=123)
		adb.fit(self.X, self.y)
		y_pred = adb.predict(self.X)
		return self.X , self.y ,y_pred


# # 画出训练数据集（用黑色表示）
# plt.scatter(X, y, c="k", s=10, label="Training Samples")
# # 画出adbr_3模型（最大迭代次数为100)的拟合效果（用蓝色表示）
# plt.plot(X, y_pred, c="b", label="n_estimators=%d" % 100, linewidth=1)
# plt.legend()
# plt.show()
