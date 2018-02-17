import sys, os.path as path
sys.path.append(path.abspath(path.join(__file__ ,"../../../")))
import numpy as np
import datetime
import pickle
from lib.MNIST import MNIST
from lib.layers import DenseLayer, ConvolutionLayer, MaxPoolingLayer, FlattenLayer, DropoutLayer, BatchNormLayer
from lib.layers import ReluLayer, SoftmaxCrossEntropyLayer
from collections import OrderedDict
from PIL import Image

def he(n_in):
    return np.sqrt(2 / n_in)


class DeepConvNet:
    def __init__(self):
        self.params = {}
        self.layers = None

        self.init_params()
        self.init_layers()

        self.is_training = False

    def init_params(self):
        self.params = {}
        self.params['W1'] = np.random.randn(3, 3, 1, 32) * he(28*28)
        self.params['b1'] = np.zeros(32)
        self.params['W2'] = np.random.randn(3, 3, 32, 32) * he(3*3*1)
        self.params['b2'] = np.zeros(32)

        self.params['W3'] = np.random.randn(3, 3, 32, 64) * he(3*3*32)
        self.params['b3'] = np.zeros(64)
        self.params['W4'] = np.random.randn(3, 3, 64, 64) * he(3*3*64)
        self.params['b4'] = np.zeros(64)

        self.params['W5'] = np.random.randn(7*7*64, 256) * he(7*7*64)
        self.params['b5'] = np.zeros(256)
        self.params['gamma1'] = np.ones(256)
        self.params['beta1'] = np.zeros(256)

        self.params['W6'] = np.random.randn(256, 128) * he(256)
        self.params['b6'] = np.zeros(128)
        self.params['gamma2'] = np.ones(128)
        self.params['beta2'] = np.zeros(128)

        self.params['W7'] = np.random.randn(128, 10) * he(128)
        self.params['b7'] = np.zeros(10)

    def init_layers(self):
        self.layers = OrderedDict()
        self.layers['Convolution1'] = ConvolutionLayer(self.params['W1'], self.params['b1'], stride=1, padding=1)
        self.layers['Relu1'] = ReluLayer()
        self.layers['Convolution2'] = ConvolutionLayer(self.params['W2'], self.params['b2'], stride=1, padding=1)
        self.layers['Relu2'] = ReluLayer()
        self.layers['MaxPooling1'] = MaxPoolingLayer(stride=2)

        self.layers['Convolution3'] = ConvolutionLayer(self.params['W3'], self.params['b3'], stride=1, padding=1)
        self.layers['Relu3'] = ReluLayer()
        self.layers['Convolution4'] = ConvolutionLayer(self.params['W4'], self.params['b4'], stride=1, padding=1)
        self.layers['Relu4'] = ReluLayer()
        self.layers['MaxPooling2'] = MaxPoolingLayer(stride=2)
        self.layers['Reshape'] = FlattenLayer()

        self.layers['Dense1'] = DenseLayer(self.params['W5'], self.params['b5'])
        self.layers['BatchNorm1'] = BatchNormLayer(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu5'] = ReluLayer()
        self.layers['Dropout1'] = DropoutLayer(num_neurons=256, keep_prob=0.5)

        self.layers['Dense2'] = DenseLayer(self.params['W6'], self.params['b6'])
        self.layers['BatchNorm2'] = BatchNormLayer(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu6'] = ReluLayer()
        self.layers['Dropout2'] = DropoutLayer(num_neurons=128, keep_prob=0.5)

        self.layers['Dense3'] = DenseLayer(self.params['W7'], self.params['b7'])
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
        for name, layer in self.layers.items():
            if name.startswith('Dropout'):
                out = layer.forward(out, is_training=self.is_training)
            else:
                out = layer.forward(out)

        return out



def ensemble_predict(nets, X):
    predictions = []
    for net in nets:
        out = net.predict(X)
        predictions.append(out)

    return np.sum(predictions, axis=0)

def ensemble_accuracy_count(X, T):
    Y = ensemble_predict(nets, X)
    Y_index = np.argmax(Y, axis=1)
    T_index = np.argmax(T, axis=1)
    return np.sum(Y_index == T_index)

if __name__ == '__main__':
    print("this is main")

    mnist = MNIST()
    train_images, train_labels, test_images, test_labels = mnist.get_dataset(normalize=True, for_conv_net=True)
    batch_size = 100

    # Ensemble from 1 to 10
    for num_nets in range(1, 11):
        print("============== Ensemble with {} networks".format(num_nets))

        # Load network
        nets = []
        for i in range(num_nets):
            net = DeepConvNet()
            param_path = path.join(path.dirname(__file__ ), 'params', 'deep_conv_solo_{}_params.pkl'.format(i))
            net.load_params(param_path)
            nets.append(net)

        log = {
            'test_acc': 0
        }

        # Calculate accuracy
        correct_prediction = 0
        for batch in range(test_images.shape[0] // batch_size):
            index_start = batch * batch_size
            index_end = (batch+1) * batch_size
            t_image = test_images[index_start:index_end, :, :, :]
            t_label = test_labels[index_start:index_end]

            co = ensemble_accuracy_count(t_image, t_label)
            print("{}-{}: {}/{}".format(index_start, index_end, co, t_image.shape[0]))

            correct_prediction += co

        test_acc = correct_prediction / test_images.shape[0]
        print("[Accuracy] test: {}".format(test_acc))

        log['test_acc'] = test_acc
        pickle.dump(log, open(path.join(path.dirname(__file__ ), 'deep_conv_ensemble_{}_log.pkl'.format(num_nets)), "wb"))
