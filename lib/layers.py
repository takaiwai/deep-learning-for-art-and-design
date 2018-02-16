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

class ConvolutionLayer:
    def __init__(self, W, b, padding=0, stride=1):
        self.W = W
        self.b = b
        self.padding = padding
        self.stride = stride

        self.X = None
        self.X_col = None
        self.W_col = None

        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        N_batch, H_in, W_in, C_in = X.shape
        H_filter, W_filter, C_in, C_out = self.W.shape

        H_out = (H_in + 2 * self.padding - H_filter) // self.stride + 1
        W_out = (W_in + 2 * self.padding - W_filter) // self.stride + 1

        if self.padding > 0:
            X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')

        X_col = np.zeros((N_batch * H_out * W_out, H_filter * W_filter * C_in))
        X_col_row_index = 0
        for n_batch in range(N_batch):  # TODO: Maybe I can remove this loop over N_batch?
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * self.stride
                    h_end = h_start + H_filter
                    w_start = w * self.stride
                    w_end = w_start + W_filter

                    X_slice = X[n_batch, h_start:h_end, w_start:w_end, :].transpose(2, 0, 1)
                    X_col[X_col_row_index, :] = X_slice.reshape(1, -1)
                    X_col_row_index += 1  # X_col_row_index = n_batch * (H_out * W_out) + h * W_out + w

        W_col = self.W.transpose(2, 0, 1, 3).reshape(-1, C_out)

        Y_col = np.dot(X_col, W_col)
        Y = Y_col.reshape(N_batch, H_out, W_out, C_out) + self.b

        self.X_col = X_col
        self.W_col = W_col

        return Y

    def backward(self, dY):
        N_batch, H_in, W_in, C_in = self.X.shape
        H_filter, W_filter, _, C_out = self.W.shape
        _, H_out, W_out, _ = dY.shape

        # dY
        dY_col = dY.reshape(-1, C_out)

        # db
        db = np.sum(dY, axis=(0, 1, 2))

        # dW
        dW_col = np.dot(self.X_col.T, dY_col)
        dW = dW_col.reshape(C_in, H_filter, W_filter, C_out).transpose(1, 2, 0, 3)

        # dX
        dX_col = np.dot(dY_col, self.W_col.T)
        dX = np.zeros((N_batch, H_in + 2 * self.padding, W_in + 2 * self.padding, C_in))
        dX_col_row_index = 0
        for n_batch in range(N_batch):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * self.stride
                    h_end = h_start + H_filter
                    w_start = w * self.stride
                    w_end = w_start + W_filter

                    dX_col_slice = dX_col[dX_col_row_index, :].reshape(C_in, H_filter, W_filter).transpose(1, 2, 0)
                    dX[n_batch, h_start:h_end, w_start:w_end, :] += dX_col_slice
                    dX_col_row_index += 1  # dX_col_row_index = n_batch * (H_out * W_out) + h * W_out + w

        if self.padding > 0:
            dX = dX[:, self.padding:-self.padding, self.padding:-self.padding, :]

        self.dW = dW
        self.db = db

        return dX

class MaxPoolingLayer:
    def __init__(self, stride):
        self.stride = stride
        self.X = None

    def forward(self, X):
        N_batch, H_in, W_in, C_in = X.shape

        H_out = H_in // self.stride
        W_out = W_in // self.stride

        Y = np.zeros((N_batch, H_out, W_out, C_in))
        for h in range(H_out):
            h_start = h * self.stride
            h_end = h_start + self.stride
            for w in range(W_out):
                w_start = w * self.stride
                w_end = w_start + self.stride
                X_slice = X[:, h_start:h_end, w_start:w_end, :]
                Y[:, h, w, :] = np.max(X_slice, axis=(1, 2))

        self.X = X

        return Y

    def backward(self, dY):
        N_batch, H_in, W_in, C_in = self.X.shape

        H_out = H_in // self.stride
        W_out = W_in // self.stride

        dX = np.zeros_like(self.X)

        for n_batch in range(N_batch):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * self.stride
                    h_end = h_start + self.stride
                    w_start = w * self.stride
                    w_end = w_start + self.stride

                    current_dY = dY[n_batch, h, w, :]

                    X_slice = self.X[n_batch, h_start:h_end, w_start:w_end, :]
                    flat_X_slice_by_channel = X_slice.transpose(2, 0, 1).reshape(C_in, -1)
                    max_index = np.argmax(flat_X_slice_by_channel, axis=1)

                    gradient = np.zeros_like(flat_X_slice_by_channel)
                    gradient[np.arange(C_in), max_index] = current_dY
                    gradient = gradient.reshape(X_slice.shape[2], X_slice.shape[0], X_slice.shape[1]).transpose(1, 2, 0)

                    dX[n_batch, h_start:h_end, w_start:w_end, :] = gradient

        return dX

class FlattenLayer:
    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(self.input_shape[0], -1)

    def backward(self, dY):
        return dY.reshape(self.input_shape)


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
        dmu = -1 * np.sum(dx_mu1 + dxmu2, axis=0)

        dx2 = 1.0 / N * np.ones((N, D)) * dmu

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
