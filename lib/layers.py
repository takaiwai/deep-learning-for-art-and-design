import sys, os
sys.path.append(os.pardir)
import numpy as np
from lib.functions import sigmoid, softmax, cross_entropy

class MatMulLayer():
    def __init__(self):
        self.X = None
        self.W = None

    def forward(self, X, W):
        Y = np.dot(X, W)
        self.X = X
        self.W = W
        return Y

    def backward(self, dY):
        dX = np.dot(dY, self.W.T)
        dW = np.dot(self.X.T, dY)
        return dX, dW

class MatAddLayer():
    def forward(self, X, b):
        Y = X + b
        return Y

    def backward(self, dY):
        dA = dY
        db = np.sum(dY, axis=0)
        return dA, db

class DenseLayer():
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.dW = None
        self.db = None

        self.mat_mul_layer = MatMulLayer()
        self.mat_add_layer = MatAddLayer()

    def forward(self, X):
        Y = self.mat_add_layer.forward(self.mat_mul_layer.forward(X, self.W), self.b)
        return Y

    def backward(self, dY):
        _, self.db = self.mat_add_layer.backward(dY)
        dX, self.dW = self.mat_mul_layer.backward(dY)
        return dX


class SigmoidLayer():
    pass

class SoftmaxCrossEntropyLayer():
    pass


if __name__ == '__main__':
    print("Testing layers.py")
    C = np.array([
        [3, 2],
        [1, 0],
        [0, 4],
        [5, 6],
    ])

    P = np.array([
        [1.5, 1.0, 2.0],
        [3.0, 2.0, 4.0]
    ])
    b = np.array([0.5, 1.0, 0.7])

    dense = DenseLayer(P, b)
    price_matrix = dense.forward(C)
    print(price_matrix)
    print(np.sum(price_matrix))
    dC = dense.backward(np.ones((4, 3)))
    print(dC)
    print(dense.dW)
    print(dense.db)
