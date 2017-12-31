import numpy as np
import datetime
import pickle
from lib.MNIST import MNIST
from lib.functions import sigmoid, softmax, cross_entropy

class BasicNet:
    def __init__(self):
        print("Created BasicNet instance")

        SIGMA = 0.01
        self.params = {}
        self.params['W1'] = np.random.randn(28*28, 32) * SIGMA
        self.params['b1'] = np.random.rand(32) * SIGMA
        self.params['W2'] = np.random.randn(32, 10) * SIGMA
        self.params['b2'] = np.random.rand(10) * SIGMA

    def predict(self, X):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        Y = softmax(Z2)

        return Y

    def loss(self, X, T):
        Y = self.predict(X)
        loss = cross_entropy(Y, T)

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

                batch_mask = np.random.choice(train_size, batch_size)
                batch_images = train_images[batch_mask]
                batch_labels = train_labels[batch_mask]
                self.gradient_descent(batch_images, batch_labels)

            if itr % 10 == 0:
                pickle_filename = "params_epoch_{}_itr_{}.pkl".format(epoch, itr)
                pickle.dump(self.params, open(pickle_filename, "wb"))
                print("Saved params at {}".format(pickle_filename))


    def gradient_descent(self, X, T):
        ETA = 0.001
        grads = basic_net.gradients(X, T)
        for param_name in ['W1', 'b1', 'W2', 'b2']:
            self.params[param_name] -= ETA * grads[param_name]

    def gradients(self, X, T):
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
    basic_net = BasicNet()

    mnist = MNIST()
    train_images, train_labels, test_images, test_labels = mnist.get_dataset()

    # prediction = basic_net.predict(test_images[:5])
    # print(prediction)
    #
    # loss = basic_net.loss(test_images[:5], test_labels[:5])
    # print(loss)

    # batch_images = train_images[:100]
    # batch_labels = train_labels[:100]
    # grad = basic_net.gradients(batch_images, batch_labels)

    basic_net.train(train_images, train_labels, epochs=1)


