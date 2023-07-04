
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
class LR():
    def __init__(self):
        #默认使用diabets数据集
        self.diabetes_X, self.diabetes_y = datasets.load_diabetes(return_X_y=True)
        # 如果return_X_y为 True，则（数据、目标）将是 pandas DataFrames 或 Series。 diabetes_X为一个ndarray,一共442行10列，diabetes_y为一个ndarray，共442行一列
        # 只用一个属性,442行1列，__len__为2，取第三列的数据，该列对应第一个特征
        self.diabetes_X = self.diabetes_X[:, np.newaxis, 2]  # 经过测试，不加np.newaxis也可以:diabetes_X = diabetes_X[:, 2]
        # 分为训练集和测试集
        self.X_train = self.diabetes_X[:-20]
        self.X_test = self.diabetes_X[-20:]
        self.y_train = self.diabetes_y[:-20]
        self.y_test = self.diabetes_y[-20:]

    def run(self):
        # 创建(线性回归)对象
        regr = linear_model.LinearRegression()
        regr.fit(self.X_train, self.y_train)  # 训练
        # 使用测试集预测
        y_pred = regr.predict(self.X_test)

        return  self.X_test, self.y_test , y_pred



