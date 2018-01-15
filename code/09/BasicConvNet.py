import sys, os
sys.path.append(os.pardir)
import numpy as np
import datetime
import pickle
from lib.MNIST import MNIST
from lib.layers import DenseLayer, SigmoidLayer, SoftmaxCrossEntropyLayer, ReluLayer
from collections import OrderedDict

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

        self.X = X
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
        # print('=================== Max Pooling forward ===================')
        # print("--------input X: ", X.shape)
        # print(X)
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
        # print("--------output Y: ", Y.shape)
        # print(Y)

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

class ReshapeLayer:
    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(self.input_shape[0], -1)

    def backward(self, dY):
        return dY.reshape(self.input_shape)

class BasicConvNet:
    def __init__(self):
        self.params = None
        self.layers = None

        self.init_params()
        self.init_layers()

    def init_params(self):
        SIGMA = 0.1

        self.params = {}
        self.params['W1'] = np.random.randn(4, 4, 1, 2) * SIGMA
        # self.params['b1'] = np.random.randn(2)
        self.params['b1'] = np.zeros(2)
        self.params['W2'] = np.random.randn(3, 3, 2, 4) * SIGMA
        # self.params['b2'] = np.random.randn(4)
        self.params['b2'] = np.zeros(4)

        # self.params['W3'] = np.random.randn(6*6*4, 8) * SIGMA
        self.params['W3'] = np.random.randn(3*3*4, 8) * SIGMA
        self.params['b3'] = np.random.rand(8) * SIGMA
        self.params['W4'] = np.random.randn(8, 10) * SIGMA
        self.params['b4'] = np.random.rand(10) * SIGMA

    def init_layers(self):
        self.layers = OrderedDict()
        self.layers['Convolution1'] = ConvolutionLayer(self.params['W1'], self.params['b1'], stride=2, padding=0)
        self.layers['Relu1'] = ReluLayer()
        self.layers['Convolution2'] = ConvolutionLayer(self.params['W2'], self.params['b2'], stride=2, padding=0)
        self.layers['Relu2'] = ReluLayer()
        self.layers['MaxPooling1'] = MaxPoolingLayer(stride=2)
        self.layers['Reshape1'] = ReshapeLayer()
        self.layers['Dense1'] = DenseLayer(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = ReluLayer()
        self.layers['Dense2'] = DenseLayer(self.params['W4'], self.params['b4'])
        self.last_layer = SoftmaxCrossEntropyLayer()

    def save_params(self, filename):
        pickle.dump(self.params, open(filename, "wb"))
        print("Saved params at {}".format(filename))

    def load_params(self, filename):
        self.params = pickle.load(open(filename, "rb"))
        print("Loaded params from {}".format(filename))
        self.init_layers()
        print("Recreated layers with the params")

    def predict(self, X):
        out = X
        for layer in self.layers.values():
            out = layer.forward(out)
        return out

    def loss(self, X, T):
        Z = self.predict(X)
        loss = self.last_layer.forward(Z, T)
        return loss

    def accuracy(self, X, T):
        Z = self.predict(X)
        Z_index = np.argmax(Z, axis=1)
        T_index = np.argmax(T, axis=1)
        return np.mean(Z_index == T_index)

    def gradient_descent(self, X, T):
        ETA = 0.001
        grads = net.gradients(X, T)
        for param_name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']:
            self.params[param_name] -= ETA * grads[param_name]

    def gradients(self, X, T):
        self.loss(X, T)

        dL = self.last_layer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dL = layer.backward(dL)

        gradients = {}
        gradients['W1'] = self.layers['Convolution1'].dW
        gradients['b1'] = self.layers['Convolution1'].db
        gradients['W2'] = self.layers['Convolution2'].dW
        gradients['b2'] = self.layers['Convolution2'].db
        gradients['W3'] = self.layers['Dense1'].dW
        gradients['b3'] = self.layers['Dense1'].db
        gradients['W4'] = self.layers['Dense2'].dW
        gradients['b4'] = self.layers['Dense2'].db

        return gradients

    def numerical_gradients(self, X, T):
        loss = lambda: self.loss(X, T)

        gradients = {}
        for param_name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']:
            print("Calculating numerical gradient with respect to: ", param_name)
            gradients[param_name] = self.numerical_gradient(loss, self.params[param_name])

        return gradients

    def numerical_gradient(self, loss, variables):
        h = 1e-7
        gradients = np.zeros_like(variables)

        itr = np.nditer(variables, flags=['multi_index'], op_flags=['readwrite'])
        while not itr.finished:
            original = itr[0].copy()

            itr[0] = original + h
            # print("original + h: {}".format(itr[0]))
            v1 = loss()
            itr[0] = original - h
            # print("original - h: {}".format(itr[0]))
            v2 = loss()
            gradients[itr.multi_index] = (v1 - v2) / (2 * h)
            # print("grad: {}".format(gradients[itr.multi_index]))

            itr[0] = original
            itr.iternext()

        return gradients

    def gradient_check(self, images, labels):
        print("Checking gradients...")
        THRESHOLD = 1e-5
        backprop_grad = self.gradients(images, labels)
        numerical_grad = self.numerical_gradients(images, labels)

        for key in backprop_grad.keys():
            b = backprop_grad[key].reshape(-1)
            n = numerical_grad[key].reshape(-1)

            diff = np.linalg.norm(b - n)
            prop = np.linalg.norm(b) + np.linalg.norm(n)
            check = diff / prop
            if check < THRESHOLD:
                result = 'OK'
            else:
                result = 'NG'

            print("gradient {}: {} ({}) diff: {}, prop: {}".format(key, result, check, diff, prop))


if __name__ == '__main__':
    print("this is main")

    np.random.seed(1)

    net = BasicConvNet()
    # fast_basic_net.load_params('params_after_5_epochs.pkl')

    mnist = MNIST()
    train_images, train_labels, test_images, test_labels = mnist.get_dataset()

    # batch_images = train_images[:20]
    # batch_labels = train_labels[:20]

    # ==== Training
    log = {
        'loss_train': [],
        'loss_train_itr': [],
        'loss_test': [],
        'loss_test_itr': [],
        'accuracy_train': [],
        'accuracy_train_itr': [],
        'accuracy_test': [],
        'accuracy_test_itr': [],
    }
    
    epochs = 5
    train_size = train_images.shape[0]
    batch_size = 100
    iteration_per_epoch = train_size // batch_size
    total_iterations = iteration_per_epoch * epochs

    # Test loss
    batch_mask = np.random.choice(train_size, 100)
    print(batch_mask)
    batch_images = train_images[batch_mask].reshape(100, 28, 28, 1)
    batch_labels = train_labels[batch_mask]
    # print("batch_images.shape: ", batch_images.shape)
    # print("batch_labels.shape: ", batch_labels.shape)
    train_loss = net.loss(batch_images, batch_labels)
    print("train_loss: ", train_loss)


    for _ in range(10):
        batch_mask = np.random.choice(train_size, 100)
        batch_images = train_images[batch_mask].reshape(100, 28, 28, 1)
        batch_labels = train_labels[batch_mask]
        net.gradient_descent(batch_images, batch_labels)


    batch_mask = np.random.choice(train_size, 1)
    # batch_mask = 42
    print(batch_mask)
    batch_images = train_images[batch_mask].reshape(1, 28, 28, 1)
    batch_labels = train_labels[batch_mask]
    print("batch_images.shape: ", batch_images.shape)
    print("batch_labels.shape: ", batch_labels.shape)

    # gradient
    print("---------- gradient ----------")
    gradient = net.gradients(batch_images, batch_labels)
    # print(gradient)

    # # numerical gradient
    # print("---------- numerical gradient ----------")
    # gradient = net.numerical_gradients(batch_images, batch_labels)
    # print(gradient)

    print("---------- gradient check ----------")
    net.gradient_check(batch_images, batch_labels)


    # itr = 0
    # for epoch in range(epochs):
    #     print("========== Epoch {} ==========".format(epoch))
    #
    #     for _ in range(iteration_per_epoch):
    #         batch_mask = np.random.choice(train_size, batch_size)
    #         batch_images = train_images[batch_mask]
    #         batch_labels = train_labels[batch_mask]
    #
    #         if itr % 100 == 0:
    #             print("Iteration {}/{}: {}".format(itr, total_iterations, datetime.datetime.now()))
    #
    #         if itr % 100 == 0:
    #             train_loss = net.loss(batch_images, batch_labels)
    #             test_loss = net.loss(test_images, test_labels)
    #             print("Losses in Iteration {}: train: {}, test: {}".format(itr, train_loss, test_loss))
    #             log['loss_train'].append(train_loss)
    #             log['loss_train_itr'].append(itr)
    #             log['loss_test'].append(test_loss)
    #             log['loss_test_itr'].append(itr)
    #
    #         if itr % 100 == 0:
    #             train_acc = net.accuracy(batch_images, batch_labels)
    #             test_acc = net.accuracy(test_images, test_labels)
    #             print("Accuracy in Iteration {}: train: {}, test: {}".format(itr, train_acc, test_acc))
    #             log['accuracy_train'].append(train_acc)
    #             log['accuracy_train_itr'].append(itr)
    #             log['accuracy_test'].append(test_acc)
    #             log['accuracy_test_itr'].append(itr)
    #
    #         # if itr % 100 == 0:
    #         #     pickle_filename = "params_epoch_{}_itr_{}.pkl".format(epoch, itr)
    #         #     fast_basic_net.save_params(pickle_filename)
    #
    #         net.gradient_descent(batch_images, batch_labels)
    #
    #         itr += 1
    #
    # print("Done!")
    # # ==== End Training
    #
    # # print(log)
    # pickle.dump(log, open('log.pkl', "wb"))
    #
    # train_loss = net.loss(train_images, train_labels)
    # test_loss = net.loss(test_images, test_labels)
    # print("[Losses] train: {}, test: {}".format(train_loss, test_loss))
    #
    # train_acc = net.accuracy(train_images, train_labels)
    # test_acc = net.accuracy(test_images, test_labels)
    # print("[Accuracy] train: {}, test: {}".format(train_acc, test_acc))
    #
    # net.save_params("params_6_epoch.pkl")
