import sys, os
sys.path.append(os.pardir)
import numpy as np
import datetime
import pickle
from lib.MNIST import MNIST
from lib.functions import sigmoid, softmax, cross_entropy
from lib.layers import DenseLayer, SigmoidLayer, SoftmaxCrossEntropyLayer
from collections import OrderedDict

class FastBasicNet:
    def __init__(self):
        print("Created FastBasicNet instance")

        SIGMA = 0.01
        self.params = {}
        self.params['W1'] = np.random.randn(28*28, 32) * SIGMA
        self.params['b1'] = np.random.rand(32) * SIGMA
        self.params['W2'] = np.random.randn(32, 10) * SIGMA
        self.params['b2'] = np.random.rand(10) * SIGMA

        self.layers = OrderedDict()
        self.layers['Dense1'] = DenseLayer(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = SigmoidLayer()
        self.layers['Dense2'] = DenseLayer(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxCrossEntropyLayer()


    def predict(self, X):
        out = X
        for layer in self.layers.values():
            out = layer.forward(out)

        return out

    def loss(self, X, T):
        Z = self.predict(X)
        loss = self.last_layer.forward(Z, T)

        return loss

    def train(self, train_images, train_labels, epochs=5):
        train_size = train_images.shape[0]
        batch_size = 100
        iteration_per_epoch = train_size // batch_size
        total_iterations = iteration_per_epoch * epochs

        for epoch in range(epochs):
            print("========== Epoch {} ==========".format(epoch))

            for itr in range(iteration_per_epoch):
                print("Iteration {}/{}: {}".format(itr, total_iterations, datetime.datetime.now()))

                if itr % 10 == 0:
                    loss = self.loss(train_images, train_labels)
                    print("Loss in Iteration {}: {}".format(itr, loss))

                if itr % 100 == 0:
                    pickle_filename = "params_epoch_{}_itr_{}.pkl".format(epoch, itr)
                    pickle.dump(self.params, open(pickle_filename, "wb"))
                    print("Saved params at {}".format(pickle_filename))

                batch_mask = np.random.choice(train_size, batch_size)
                batch_images = train_images[batch_mask]
                batch_labels = train_labels[batch_mask]
                self.gradient_descent(batch_images, batch_labels)

        pickle_filename = "params_after_{}_epochs.pkl".format(epochs)
        pickle.dump(self.params, open(pickle_filename, "wb"))
        print("Saved params at {}".format(pickle_filename))


    def gradient_descent(self, X, T):
        ETA = 0.1
        grads = fast_basic_net.numerical_gradients(X, T)
        for param_name in ['W1', 'b1', 'W2', 'b2']:
            self.params[param_name] -= ETA * grads[param_name]

    def gradients(self, X, T):
        self.loss(X, T)

        dL = self.last_layer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        print("layers:", layers)
        for layer in layers:
            dL = layer.backward(dL)

        gradients = {}
        gradients['W1'] = self.layers['Dense1'].dW
        gradients['b1'] = self.layers['Dense1'].db
        gradients['W2'] = self.layers['Dense2'].dW
        gradients['b2'] = self.layers['Dense2'].db

        return gradients

    def numerical_gradients(self, X, T):
        loss = lambda: self.loss(X, T)

        gradients = {}
        for param_name in ['W1', 'b1', 'W2', 'b2']:
            gradients[param_name] = self.numerical_gradient(loss, self.params[param_name])

        return gradients

    def numerical_gradient(self, loss, variables):
        h = 1e-4
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


if __name__ == '__main__':
    print("this is main")
    fast_basic_net = FastBasicNet()

    mnist = MNIST()
    train_images, train_labels, test_images, test_labels = mnist.get_dataset()

    # prediction = fast_basic_net.predict(test_images[:5])
    # print(prediction)
    #
    # loss = fast_basic_net.loss(test_images[:5], test_labels[:5])
    # print(loss)

    batch_images = train_images[:100]
    batch_labels = train_labels[:100]
    grad = fast_basic_net.gradients(batch_images, batch_labels)
    print(grad)

    # fast_basic_net.train(train_images, train_labels, epochs=5)
    # print("Done!")

