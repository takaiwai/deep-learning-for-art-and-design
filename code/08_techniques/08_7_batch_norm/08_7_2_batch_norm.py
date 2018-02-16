import sys, os.path as path
sys.path.append(path.abspath(path.join(__file__ ,"../../../..")))
import numpy as np
import pickle
from lib.MNIST import MNIST
from lib.layers import DenseLayer, BatchNormLayer, ReluLayer, SoftmaxCrossEntropyLayer, SigmoidLayer
from collections import OrderedDict

def he(n_in):
    return np.sqrt(2 / n_in)

class SevenLayerNet:
    def __init__(self):
        self.params = None
        self.layers = None

        self.init_params()
        self.init_layers()

    def init_params(self):
        self.params = {}
        self.params['W1'] = np.random.randn(28*28, 64) * he(28*28)
        self.params['b1'] = np.random.randn(64) * he(28*28)
        self.params['gamma1'] = np.ones(64)
        self.params['beta1'] = np.zeros(64)

        self.params['W2'] = np.random.randn(64, 50) * he(64)
        self.params['b2'] = np.random.randn(50) * he(64)
        self.params['gamma2'] = np.ones(50)
        self.params['beta2'] = np.zeros(50)

        self.params['W3'] = np.random.randn(50, 40) * he(50)
        self.params['b3'] = np.zeros(40)
        self.params['gamma3'] = np.ones(40)
        self.params['beta3'] = np.zeros(40)
        
        self.params['W4'] = np.random.randn(40, 30) * he(40)
        self.params['b4'] = np.zeros(30)
        self.params['gamma4'] = np.ones(30)
        self.params['beta4'] = np.zeros(30)
        
        self.params['W5'] = np.random.randn(30, 30) * he(30)
        self.params['b5'] = np.zeros(30)
        self.params['gamma5'] = np.ones(30)
        self.params['beta5'] = np.zeros(30)
        
        self.params['W6'] = np.random.randn(30, 30) * he(30)
        self.params['b6'] = np.zeros(30)
        self.params['gamma6'] = np.ones(30)
        self.params['beta6'] = np.zeros(30)
        
        self.params['W7'] = np.random.randn(30, 10) * he(30)
        self.params['b7'] = np.zeros(10)

    def init_layers(self):
        self.layers = OrderedDict()
        self.layers['Dense1'] = DenseLayer(self.params['W1'], self.params['b1'])
        self.layers['BatchNorm1'] = BatchNormLayer(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = SigmoidLayer()

        self.layers['Dense2'] = DenseLayer(self.params['W2'], self.params['b2'])
        self.layers['BatchNorm2'] = BatchNormLayer(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = SigmoidLayer()

        self.layers['Dense3'] = DenseLayer(self.params['W3'], self.params['b3'])
        self.layers['BatchNorm3'] = BatchNormLayer(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = SigmoidLayer()
        
        self.layers['Dense4'] = DenseLayer(self.params['W4'], self.params['b4'])
        self.layers['BatchNorm4'] = BatchNormLayer(self.params['gamma4'], self.params['beta4'])
        self.layers['Relu4'] = SigmoidLayer()
        
        self.layers['Dense5'] = DenseLayer(self.params['W5'], self.params['b5'])
        self.layers['BatchNorm5'] = BatchNormLayer(self.params['gamma5'], self.params['beta5'])
        self.layers['Relu5'] = SigmoidLayer()
        
        self.layers['Dense6'] = DenseLayer(self.params['W6'], self.params['b6'])
        self.layers['BatchNorm6'] = BatchNormLayer(self.params['gamma6'], self.params['beta6'])
        self.layers['Relu6'] = SigmoidLayer()
        
        self.layers['Dense7'] = DenseLayer(self.params['W7'], self.params['b7'])
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
        ETA = 0.1
        grads = seven_layer_net.gradients(X, T)
        for param_name in list(self.params.keys()):
            self.params[param_name] -= ETA * grads[param_name]

    def gradients(self, X, T):
        self.loss(X, T)

        dL = self.last_layer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dL = layer.backward(dL)

        gradients = {}
        gradients['W1'] = self.layers['Dense1'].dW
        gradients['b1'] = self.layers['Dense1'].db

        gradients['gamma1'] = self.layers['BatchNorm1'].dgamma
        gradients['beta1'] = self.layers['BatchNorm1'].dbeta

        gradients['W2'] = self.layers['Dense2'].dW
        gradients['b2'] = self.layers['Dense2'].db
        gradients['gamma2'] = self.layers['BatchNorm2'].dgamma
        gradients['beta2'] = self.layers['BatchNorm2'].dbeta

        gradients['W3'] = self.layers['Dense3'].dW
        gradients['b3'] = self.layers['Dense3'].db
        gradients['gamma3'] = self.layers['BatchNorm3'].dgamma
        gradients['beta3'] = self.layers['BatchNorm3'].dbeta
        
        gradients['W4'] = self.layers['Dense4'].dW
        gradients['b4'] = self.layers['Dense4'].db
        gradients['gamma4'] = self.layers['BatchNorm4'].dgamma
        gradients['beta4'] = self.layers['BatchNorm4'].dbeta
        
        gradients['W5'] = self.layers['Dense5'].dW
        gradients['b5'] = self.layers['Dense5'].db
        gradients['gamma5'] = self.layers['BatchNorm5'].dgamma
        gradients['beta5'] = self.layers['BatchNorm5'].dbeta
        
        gradients['W6'] = self.layers['Dense6'].dW
        gradients['b6'] = self.layers['Dense6'].db
        gradients['gamma6'] = self.layers['BatchNorm6'].dgamma
        gradients['beta6'] = self.layers['BatchNorm6'].dbeta
        
        gradients['W7'] = self.layers['Dense7'].dW
        gradients['b7'] = self.layers['Dense7'].db

        return gradients

    def numerical_gradients(self, X, T):
        loss = lambda: self.loss(X, T)

        gradients = {}
        for param_name in list(self.params.keys()):
            print("calculating: ", param_name)
            gradients[param_name] = self.numerical_gradient(loss, self.params[param_name])

        return gradients

    def numerical_gradient(self, loss, variables):
        h = 1e-4
        gradients = np.zeros_like(variables)

        itr = np.nditer(variables, flags=['multi_index'], op_flags=['readwrite'])
        while not itr.finished:
            original = itr[0].copy()

            itr[0] = original + h
            v1 = loss()
            itr[0] = original - h
            v2 = loss()
            gradients[itr.multi_index] = (v1 - v2) / (2 * h)

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

            print("========== {}".format(key))
            print(diff)
            print(prop)
            print("gradient {}: {} ({})".format(key, result, check))


if __name__ == '__main__':
    print("this is main")

    np.random.seed(1229)

    seven_layer_net = SevenLayerNet()
    # fast_basic_net.load_params('params_after_5_epochs.pkl')

    mnist = MNIST()
    train_images, train_labels, test_images, test_labels = mnist.get_dataset()

    # batch_images = train_images[:20]
    # batch_labels = train_labels[:20]

    # ==== Training
    log = {
        'accuracy_train': [],
        'accuracy_train_itr': [],
        'accuracy_test': [],
        'accuracy_test_itr': []
    }
    
    epochs = 0
    train_size = train_images.shape[0]
    batch_size = 100
    iteration_per_epoch = train_size // batch_size
    total_iterations = iteration_per_epoch * epochs

    itr = 0
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))


        seven_layer_net.is_training = True
        for _ in range(iteration_per_epoch):

            if itr % 60 == 0:
                seven_layer_net.is_training = False
                train_acc = seven_layer_net.accuracy(train_images, train_labels)
                test_acc = seven_layer_net.accuracy(test_images, test_labels)
                log['accuracy_train'].append(train_acc)
                log['accuracy_train_itr'].append(itr)
                log['accuracy_test'].append(test_acc)
                log['accuracy_test_itr'].append(itr)
                print("[Accuracy] train: {}, test: {}".format(train_acc, test_acc))
                seven_layer_net.is_training = True

            batch_mask = np.random.choice(train_size, batch_size)
            batch_images = train_images[batch_mask]
            batch_labels = train_labels[batch_mask]

            seven_layer_net.gradient_descent(batch_images, batch_labels)
            itr += 1

        seven_layer_net.is_training = False

    print("Done!")

    seven_layer_net.gradient_check(train_images[:50, :], train_labels[:50])

    train_acc = seven_layer_net.accuracy(train_images, train_labels)
    test_acc = seven_layer_net.accuracy(test_images, test_labels)
    log['accuracy_train'].append(train_acc)
    log['accuracy_train_itr'].append(itr)
    log['accuracy_test'].append(test_acc)
    log['accuracy_test_itr'].append(itr)
    print("[Accuracy] train: {}, test: {}".format(train_acc, test_acc))

    # pickle.dump(log, open(path.join(path.dirname(__file__ ), '08_7_2_.pkl'), "wb"))
