import numpy as np
from lib.MNIST import MNIST

class BasicNet:
    def __init__(self):
        print("Created BasicNet instance")
        self.mnist = MNIST()

        train_images, train_labels, test_images, test_labels = self.mnist.get_dataset()
        print(train_images.shape)
        print(train_labels.shape)
        print(test_images.shape)
        print(test_labels.shape)

        self.mnist.show(train_images[0], train_labels[0])
        self.mnist.show(train_images[10], train_labels[10])
        self.mnist.show(test_images[0], test_labels[0])
        self.mnist.show(test_images[10], test_labels[10])



if __name__ == '__main__':
    print("this is main")
    basic_net = BasicNet()
