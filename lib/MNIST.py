import numpy as np
import matplotlib.pyplot as plt
import os.path
from urllib import request
import gzip
import pickle

class MNIST:
    BASE_URL = 'http://yann.lecun.com/exdb/mnist/'
    FILE_NAMES = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
    ]
    TMP_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../tmp/'
    PICKLE_PATH = TMP_DIR + 'mnist.pkl'

    def __init__(self):
        print("This is MNIST class")

        self.dataset = None

        if os.path.exists(self.PICKLE_PATH):
            self.load_pickle()
        else:
            self.download_binaries()
            self.create_pickle()

    def get_dataset(self, normalize=True, for_conv_net=False):
        train_images = self.dataset['train_images'].astype(np.float32)
        train_labels = np.eye(10)[self.dataset['train_labels']]
        test_images = self.dataset['test_images'].astype(np.float32)
        test_labels = np.eye(10)[self.dataset['test_labels']]

        if normalize:
            train_images /= 255.0
            test_images /= 255.0

        if for_conv_net:
            train_images = train_images.reshape(-1, 1, 28, 28)
            test_images = test_images.reshape(-1, 1, 28, 28)

        return train_images, train_labels, test_images, test_labels

    def show(self, image, label):
        print("Digit: {}, Raw Label: {}".format(np.argmax(label), label))
        plt.gray()
        plt.imshow(image.reshape(28,28))
        plt.show()


    def load_pickle(self):
        self.dataset = pickle.load(open(self.PICKLE_PATH, "rb"))
        print("Loaded MNIST pickle from {}".format(self.PICKLE_PATH))

    def create_pickle(self):
        self.dataset = {
            'train_images': self.load_images(self.FILE_NAMES[0]),
            'train_labels': self.load_labels(self.FILE_NAMES[1]),
            'test_images': self.load_images(self.FILE_NAMES[2]),
            'test_labels': self.load_labels(self.FILE_NAMES[3]),
        }
        pickle.dump(self.dataset, open(self.PICKLE_PATH, "wb"))

        print("Created MNIST pickle at {}".format(self.PICKLE_PATH))

    def download_binaries(self):
        for filename in self.FILE_NAMES:
            print("Downloading {} ...".format(filename))
            request.urlretrieve(self.BASE_URL + filename, self.TMP_DIR + filename)

        print("Downloaded all the files")

    def load_images(self, filename):
        with gzip.open(self.TMP_DIR + filename, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

        return images

    def load_labels(self, filename):
        with gzip.open(self.TMP_DIR + filename, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return labels