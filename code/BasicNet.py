import numpy as np
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


if __name__ == '__main__':
    print("this is main")
    basic_net = BasicNet()

    mnist = MNIST()
    train_images, train_labels, test_images, test_labels = mnist.get_dataset()

    prediction = basic_net.predict(test_images[:5])
    print(prediction)

    loss = basic_net.loss(test_images[:5], test_labels[:5])
    print(loss)


