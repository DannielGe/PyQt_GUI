# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MLshow(object):
    def setupUi(self, MLshow):
        MLshow.setObjectName("MLshow")
        MLshow.resize(757, 652)
        self.line = QtWidgets.QFrame(MLshow)
        self.line.setGeometry(QtCore.QRect(20, 80, 731, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(MLshow)
        self.label.setGeometry(QtCore.QRect(230, 20, 301, 51))
        self.label.setStyleSheet("font: 20pt \"Adobe 黑体 Std R\";\n"
"font: 36pt \"Adobe 黑体 Std R\";")
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(MLshow)
        self.comboBox.setGeometry(QtCore.QRect(19, 160, 150, 41))
        self.comboBox.setStyleSheet("font: 25 12pt \"Adobe Song Std\";\n"
"font: 25 18pt \"Adobe Song Std\";")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.principleButton = QtWidgets.QPushButton(MLshow)
        self.principleButton.setGeometry(QtCore.QRect(50, 440, 91, 31))
        self.principleButton.setStyleSheet("font: 25 14pt \"Adobe 宋体 Std L\";")
        self.principleButton.setObjectName("principleButton")
        self.showButton = QtWidgets.QPushButton(MLshow)
        self.showButton.setGeometry(QtCore.QRect(50, 540, 91, 31))
        self.showButton.setStyleSheet("font: 25 14pt \"Adobe 宋体 Std L\";")
        self.showButton.setObjectName("showButton")
        self.line_2 = QtWidgets.QFrame(MLshow)
        self.line_2.setGeometry(QtCore.QRect(170, 110, 31, 511))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.runButton = QtWidgets.QPushButton(MLshow)
        self.runButton.setGeometry(QtCore.QRect(50, 490, 91, 31))
        self.runButton.setStyleSheet("font: 25 14pt \"Adobe 宋体 Std L\";")
        self.runButton.setObjectName("runButton")
        self.stackedWidget = QtWidgets.QStackedWidget(MLshow)
        self.stackedWidget.setGeometry(QtCore.QRect(210, 110, 511, 501))
        self.stackedWidget.setObjectName("stackedWidget")
        self.prinStack = QtWidgets.QWidget()
        self.prinStack.setObjectName("prinStack")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.prinStack)
        self.plainTextEdit.setGeometry(QtCore.QRect(0, 0, 511, 491))
        self.plainTextEdit.setStyleSheet("font: 25 14pt \"Adobe 宋体 Std L\";")
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.stackedWidget.addWidget(self.prinStack)
        self.runStack = QtWidgets.QWidget()
        self.runStack.setObjectName("runStack")
        self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(self.runStack)
        self.plainTextEdit_2.setGeometry(QtCore.QRect(0, 0, 511, 481))
        self.plainTextEdit_2.setStyleSheet("font: 25 14pt \"Adobe 宋体 Std L\";")
        self.plainTextEdit_2.setObjectName("plainTextEdit_2")
        self.stackedWidget.addWidget(self.runStack)
        self.resultStack = QtWidgets.QWidget()
        self.resultStack.setObjectName("resultStack")
        self.stackedWidget.addWidget(self.resultStack)
        self.clearButton = QtWidgets.QPushButton(MLshow)
        self.clearButton.setGeometry(QtCore.QRect(50, 590, 91, 31))
        self.clearButton.setStyleSheet("font: 25 14pt \"Adobe 宋体 Std L\";")
        self.clearButton.setObjectName("clearButton")

        self.retranslateUi(MLshow)
        self.stackedWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MLshow)

    def retranslateUi(self, MLshow):
        _translate = QtCore.QCoreApplication.translate
        MLshow.setWindowTitle(_translate("MLshow", "机器学习展示"))
        self.label.setText(_translate("MLshow", "机器学习展示"))
        self.comboBox.setItemText(0, _translate("MLshow", "线性回归"))
        self.comboBox.setItemText(1, _translate("MLshow", "朴素贝叶斯"))
        self.comboBox.setItemText(2, _translate("MLshow", "决策树"))
        self.comboBox.setItemText(3, _translate("MLshow", "随机森林"))
        self.comboBox.setItemText(4, _translate("MLshow", "支持向量机"))
        self.comboBox.setItemText(5, _translate("MLshow", "AdaBoost"))
        self.comboBox.setItemText(6, _translate("MLshow", "K-means"))
        self.comboBox.setItemText(7, _translate("MLshow", "KNN"))
        self.principleButton.setText(_translate("MLshow", "原理"))
        self.showButton.setText(_translate("MLshow", "效果"))
        self.runButton.setText(_translate("MLshow", "运行"))
        self.clearButton.setText(_translate("MLshow", "清空"))
