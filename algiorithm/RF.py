# 随机森林回归
import matplotlib as mpl
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import ExtraTreesRegressor

class RF():
    def __init__(self):
        # 产生心状坐标
        t = np.arange(0, 2 * np.pi, 0.1)
        x = 16 * np.sin(t) ** 3
        self.X = x[:, np.newaxis]
        self.y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        self.y[::7] += 3 * (1 - np.random.rand(9))  # 增加噪声，在每数2个数的时候增加一点噪声

    def run(self):
        rf = RandomForestRegressor(n_estimators=100)  # 一般来说n_estimators越大越好，运行结果呈现出的两种结果该值分别是10和1000
        y_pred=rf.fit(self.X, self.y).predict(self.X)
        return self.X ,self.y, y_pred


if __name__ == '__main__':
    rf = RF()
    x, y , y_pred = rf.run()
    plt.scatter(x,y ,color='red' ,label = "真实值")
    plt.plot(x,y_pred ,color='blue' ,label = "预测值")
    plt.title("随机森林")
    plt.legend()
    plt.show()
