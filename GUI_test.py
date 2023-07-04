from PyQt5.QtWidgets import *
from window import Ui_MLshow

from algiorithm.LR import *
from algiorithm.NavieBayes import *
from algiorithm.DT import *
from algiorithm.RF import *
from algiorithm.SVM import *
from algiorithm.AdaBoost import *
from algiorithm.KMeans import *
from algiorithm.KNN import *


from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体（解决中文无法显示的问题）
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号“-”显示方块的问题

import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
#画布用来展示效果图
#创建一个matplotlib图形绘制类
class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        #第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.fig) #此句必不可少，否则不能显示图形
        #第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)


#业务逻辑

class Work(QWidget ,Ui_MLshow):  # 继承于UI父类
    def __init__(self):
        super(Work, self).__init__()
        self.setupUi(self)

        # 点击原理展示按钮 跳转到原理展示页面
        self.principleButton.clicked.connect(self.dispaly_principle)
        # 点击运行展示按钮 跳转到运行展示页面
        self.runButton.clicked.connect(self.dispaly_run)
        # 点击结果展示按钮 跳转到结果展示页面
        self.showButton.clicked.connect(self.dispaly_result)
        # 点击清除按钮清空展示
        self.clearButton.clicked.connect(self.ReSet)


    def Plot(self):  # 这里是绘图的关键
        self.F = MyFigure(width=3, height=2, dpi=100)  # 创建实例
        self.plot_FigureLayout = QGridLayout(self.resultStack)  # 利用栅格布局将图像与画板连接
        self.plot_FigureLayout.addWidget(self.F)

    def ReSet(self):
        self.F.deleteLater()  # 删除图像对象
        self.plot_FigureLayout.deleteLater()  # 删除布局

    # # 三个堆栈对应的页面展示
    def dispaly_principle(self):
        method_name = ["LR","NavieBayes","DT","RF","SVM","AdaBoost","KMeans","KNN"]
        with open("E:/PycharmProjects/MLwork/algiorithm/"+method_name[self.comboBox.currentIndex()], "r",encoding="UTF-8") as f:  # 打开文件
            self.principle = f.read()  # 读取文件
        self.plainTextEdit.setPlainText("第{}个算法————{}的原理"
                                         .format((self.comboBox.currentIndex()+1),self.comboBox.currentText())+'\n'+self.principle )
        self.stackedWidget.setCurrentIndex(0)

    def dispaly_run(self):
        self.plainTextEdit_2.setPlainText("第{}个算法————{}正在运行，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，"
                                          .format((self.comboBox.currentIndex() + 1), self.comboBox.currentText()))

        if self.comboBox.currentIndex() == 0: #调用Lr
            lr = LR()
            self.X ,self.y ,self.y_pred = lr.run()

        if self.comboBox.currentIndex() == 1:  # 调用NB
            nb = NB()
            self.y , self.y_pred = nb.run()

        if self.comboBox.currentIndex() == 2:  # 调用DT
            dt = DT()
            self.y , self.y_pred = dt.run()

        if self.comboBox.currentIndex() == 3:  # 调用RF
            rf = RF()
            self.X ,self.y ,self.y_pred = rf.run()

        if self.comboBox.currentIndex() == 4: #调用SVM
            svm = SVM()
            self.xx,self.yy ,self.Z, self.X ,self.y  = svm.run()

        if self.comboBox.currentIndex() == 5:  # 调用ADB
            adb = ADB()
            self.X, self.y, self.y_pred = adb.run()

        if self.comboBox.currentIndex() == 6:  # 调用KMeans
            km = KM()
            self.X , self.y_pred = km.run()

        if self.comboBox.currentIndex() == 7:  # 调用KNN
            knn = KNN()
            self.y ,self.y_pred = knn.run()


        self.plainTextEdit_2.setPlainText("第{}个算法————{},运行完毕，请查看效果!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                                          .format((self.comboBox.currentIndex() + 1), self.comboBox.currentText()))

        self.stackedWidget.setCurrentIndex(1)

    def dispaly_result(self):

        if self.comboBox.currentIndex() == 0:  # 调用Lr
            self.draw0(self.X ,self.y ,self.y_pred)

        if self.comboBox.currentIndex() == 1: #调用NB
            self.draw1( self.y, self.y_pred)

        if self.comboBox.currentIndex() == 2: #调用DT
            self.draw1( self.y, self.y_pred)

        if self.comboBox.currentIndex() == 3: #调用RF
            self.draw0(self.X ,self.y ,self.y_pred)

        if self.comboBox.currentIndex() == 4:  # 调用SVM
            self.draw4(self.xx,self.yy ,self.Z, self.X ,self.y)

        if self.comboBox.currentIndex() == 5: #调用ADB
            self.draw0(self.X ,self.y ,self.y_pred)

        if self.comboBox.currentIndex() == 6: #调用KMeans
            self.draw6( self.X, self.y_pred)

        if self.comboBox.currentIndex() == 7: #调用KNN
            self.draw1(self.y ,self.y_pred)

        self.stackedWidget.setCurrentIndex(2)

    def draw0(self,X ,y ,y_pred):
        self.Plot()
        self.F.axes.scatter(X, y, color='red',label="真实值")
        self.F.axes.plot(X, y_pred, color='blue', linewidth=3,label = "预测值")
        self.F.axes.legend()
        self.F.fig.suptitle(self.comboBox.currentText())

    def draw1(self,y ,y_pred):
        self.Plot()
        X = np.arange(y.shape[0])
        self.F.axes.scatter(X, y, color='red',label="真实值")
        self.F.axes.scatter(X, y_pred, color='blue',label = "预测值")
        self.F.axes.legend()
        self.F.fig.suptitle(self.comboBox.currentText())

    def draw4(self, xx, yy, Z ,X ,y):
        self.Plot()
        self.F.axes.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        self.F.axes.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        self.F.fig.suptitle(self.comboBox.currentText())

    def draw6(self, X,y_pred):
        self.Plot()
        self.F.axes.scatter(X[:, 0], X[:, 1], c=y_pred)
        self.F.fig.suptitle(self.comboBox.currentText())



