from PyQt6 import QtGui, QtCore

def body_brush():
    grad = QtGui.QLinearGradient(0,-100,0,20)
    grad.setColorAt(0.0, QtGui.QColor(40,40,40,240))
    grad.setColorAt(1.0, QtGui.QColor(200,200,200,60))
    return QtGui.QBrush(grad)

def belly_brush():
    return QtGui.QBrush(QtGui.QColor(230,230,230,40))

