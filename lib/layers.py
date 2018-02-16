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
    def __init__(self):
        self.A = None

    def forward(self, Z):
        self.A = sigmoid(Z)
        return self.A

    def backward(self, dA):
        dZ = dA * (1 - self.A) * self.A
        return dZ

class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = (X <= 0)
        Y = X.copy()
        Y[self.mask] = 0

        return Y

    def backward(self, dY):
        dY[self.mask] = 0
        dX = dY

        return dX

class DropoutLayer:
    def __init__(self, num_neurons, keep_prob=0.8):
        self.num_neurons = num_neurons
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, X, is_training=False):
        if is_training:
            self.mask = np.random.rand(X.shape[0], self.num_neurons) < self.keep_prob
            return X * self.mask / self.keep_prob
        else:
            return X

    def backward(self, dY):
        return dY * self.mask / self.keep_prob

class BatchNormLayer:
    EPSILON = 1e-8
    def __init__(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta

        self.dgamma = None
        self.dbeta = None

        self.cache = None

    def forward(self, X):
        N, D = X.shape

        # if is_training:
        mu = 1.0 / N * np.sum(X, axis=0)
        x_mu = X - mu
        sq = x_mu ** 2

        var = 1.0 / N * np.sum(sq, axis=0)
        sqrt_var = np.sqrt(var + BatchNormLayer.EPSILON)
        inv_var = 1.0 / sqrt_var
        x_hat = x_mu * inv_var
        gamma_x = self.gamma * x_hat
        Y = gamma_x + self.beta

        self.cache = (x_hat, x_mu, inv_var, sqrt_var, var)

        return Y

    def backward(self, dY):
        x_hat, x_mu, inv_var, sqrt_var, var = self.cache

        N, D = dY.shape

        self.dbeta = np.sum(dY, axis=0)
        dgammax = dY

        self.dgamma = np.sum(dgammax * x_hat, axis=0)
        dx_hat = dgammax * self.gamma

        dinv_var = np.sum(dx_hat * x_mu, axis=0)
        dx_mu1 = dx_hat * inv_var

        dsqrt_var = -1.0 / (sqrt_var ** 2) * dinv_var

        dvar = 0.5 / np.sqrt(var + BatchNormLayer.EPSILON) * dsqrt_var

        dsq = 1.0 / N * np.ones((N, D)) * dvar

        dxmu2 = 2 * x_mu * dsq

        dx1 = (dx_mu1 + dxmu2)
        dmu = -1 * np.sum(dx_mu1+dxmu2, axis=0)

        dx2 = -1 / N * np.ones((N, D)) * dmu

        dx = dx1 + dx2

        return dx

class SoftmaxCrossEntropyLayer():
    def __init__(self):
        self.Y = None
        self.T = None
        self.loss = None

    def forward(self, Z, T):
        self.Y = softmax(Z)
        self.T = T
        self.loss = cross_entropy(self.Y, T)

        # print("Y")
        # print(self.Y)
        # print("T")
        # print(T)
        # print("loss")
        # print(self.loss)

        return self.loss

    def backward(self, dL=1.0):
        batch_size = self.Y.shape[0]

        # dZ = (self.Y - self.T) * dL / batch_size
        dZ = (self.Y - self.T) / batch_size
        # print("dZ")
        # print(dZ)
        return dZ


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

    # dense = DenseLayer(P, b)
    # price_matrix = dense.forward(C)
    # print(price_matrix)
    # print(np.sum(price_matrix))
    # dC = dense.backward(np.ones((4, 3)))
    # print(dC)
    # print(dense.dW)
    # print(dense.db)

    # sigmoid_layer = SigmoidLayer()
    # activation = sigmoid_layer.forward(P)
    # print("Activation")
    # print(activation)
    # dA = sigmoid_layer.backward(np.ones((2, 3)))
    # print("dA")
    # print(dA)

    L = np.array([
        [0, 0, 1],
        [0, 1, 0]
    ])

    softmax_cross_entropy_layer = SoftmaxCrossEntropyLayer()
    loss = softmax_cross_entropy_layer.forward(P, L)
    dZ = softmax_cross_entropy_layer.backward()

    print("other example")

    P2 = np.array([
        [1.5, 1.0, 2.0],
        [3.0, 2.0, 4.0],
        [-3.0, 1.0, -0.6]
    ])
    L2 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0]
    ])
    softmax_cross_entropy_layer = SoftmaxCrossEntropyLayer()
    loss = softmax_cross_entropy_layer.forward(P2, L2)
    dZ = softmax_cross_entropy_layer.backward()
