import sys, os.path as path
sys.path.append(path.abspath(path.join(__file__ ,"../../../..")))
import numpy as np
import pickle
from lib.MNIST import MNIST
from lib.layers import DenseLayer, ReluLayer, SoftmaxCrossEntropyLayer
from collections import OrderedDict

def he(n_in):
    return np.sqrt(2 / n_in)

class AdamOptimizer:
    def __init__(self, params, ETA=0.001, BETA1 = 0.9, BETA2 = 0.999, EPSILON = 1e-8):
        self.iteration = 0
        self.v = {}
        self.s = {}
        self.v_corrected = {}
        self.s_corrected = {}

        for key, param in params.items():
            self.v[key] = np.zeros_like(param)
            self.s[key] = np.zeros_like(param)
            self.v_corrected[key] = np.zeros_like(param)
            self.s_corrected[key] = np.zeros_like(param)

        self.ETA = ETA
        self.BETA1 = BETA1
        self.BETA2 = BETA2
        self.EPSILON = EPSILON

    def update(self, params, grads):
        c_beta1 = self.BETA1 ** (self.iteration + 1)
        c_beta2 = self.BETA2 ** (self.iteration + 1)
        self.iteration += 1

        for key in self.v.keys():
            self.v[key] = self.BETA1 * self.v[key] + (1 - self.BETA1) * grads[key]
            self.v_corrected[key] = self.v[key] / (1 - c_beta1)

        for key in self.s.keys():
            self.s[key] = self.BETA2 * self.s[key] + (1 - self.BETA2) * (grads[key] ** 2)
            self.s_corrected[key] = self.s[key] / (1 - c_beta2)

        for key in params.keys():
            params[key] -= self.ETA * self.v_corrected[key] / np.sqrt(self.s_corrected[key] + self.EPSILON)



class FastBasicNet:
    def __init__(self):
        self.params = {}
        self.layers = None

        self.init_params()
        self.init_layers()

        self.optimizer = AdamOptimizer(self.params, 0.001)

    def init_params(self):
        self.params['W1'] = np.random.randn(28*28, 64) * he(28*28)
        self.params['b1'] = np.zeros(64)
        self.params['W2'] = np.random.randn(64, 50) * he(64)
        self.params['b2'] = np.zeros(50)
        self.params['W3'] = np.random.randn(50, 40) * he(50)
        self.params['b3'] = np.zeros(40)
        self.params['W4'] = np.random.randn(40, 30) * he(40)
        self.params['b4'] = np.zeros(30)
        self.params['W5'] = np.random.randn(30, 30) * he(30)
        self.params['b5'] = np.zeros(30)
        self.params['W6'] = np.random.randn(30, 30) * he(30)
        self.params['b6'] = np.zeros(30)
        self.params['W7'] = np.random.randn(30, 10) * he(30)
        self.params['b7'] = np.zeros(10)

    def init_layers(self):
        self.layers = OrderedDict()
        self.layers['Dense1'] = DenseLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Dense2'] = DenseLayer(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = ReluLayer()
        self.layers['Dense3'] = DenseLayer(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = ReluLayer()
        self.layers['Dense4'] = DenseLayer(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = ReluLayer()
        self.layers['Dense5'] = DenseLayer(self.params['W5'], self.params['b5'])
        self.layers['Relu5'] = ReluLayer()
        self.layers['Dense6'] = DenseLayer(self.params['W6'], self.params['b6'])
        self.layers['Relu6'] = ReluLayer()
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
        grads = fast_basic_net.gradients(X, T)
        self.optimizer.update(self.params, grads)

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
        gradients['W2'] = self.layers['Dense2'].dW
        gradients['b2'] = self.layers['Dense2'].db
        gradients['W3'] = self.layers['Dense3'].dW
        gradients['b3'] = self.layers['Dense3'].db
        gradients['W4'] = self.layers['Dense4'].dW
        gradients['b4'] = self.layers['Dense4'].db
        gradients['W5'] = self.layers['Dense5'].dW
        gradients['b5'] = self.layers['Dense5'].db
        gradients['W6'] = self.layers['Dense6'].dW
        gradients['b6'] = self.layers['Dense6'].db
        gradients['W7'] = self.layers['Dense7'].dW
        gradients['b7'] = self.layers['Dense7'].db

        return gradients

    def numerical_gradients(self, X, T):
        loss = lambda: self.loss(X, T)

        gradients = {}
        for param_name in list(self.params.keys()):
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

            print("gradient {}: {} ({})".format(key, result, check))


if __name__ == '__main__':
    print("this is main")

    np.random.seed(1229)

    fast_basic_net = FastBasicNet()
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
        'accuracy_train': 0,
        'accuracy_test': 0
    }
    
    epochs = 10
    train_size = train_images.shape[0]
    batch_size = 100
    iteration_per_epoch = train_size // batch_size
    total_iterations = iteration_per_epoch * epochs

    itr = 0
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))

        for _ in range(iteration_per_epoch):
            batch_mask = np.random.choice(train_size, batch_size)
            batch_images = train_images[batch_mask]
            batch_labels = train_labels[batch_mask]

            if itr % 100 == 0:
                train_loss = fast_basic_net.loss(batch_images, batch_labels)
                test_loss = fast_basic_net.loss(test_images, test_labels)
                log['loss_train'].append(train_loss)
                log['loss_train_itr'].append(itr)
                log['loss_test'].append(test_loss)
                log['loss_test_itr'].append(itr)

            # if itr % 100 == 0:
            #     pickle_filename = "params_epoch_{}_itr_{}.pkl".format(epoch, itr)
            #     fast_basic_net.save_params(pickle_filename)

            fast_basic_net.gradient_descent(batch_images, batch_labels)

            itr += 1


    print("Done!")
    # ==== End Training


    train_loss = fast_basic_net.loss(train_images, train_labels)
    test_loss = fast_basic_net.loss(test_images, test_labels)
    print("[Losses] train: {}, test: {}".format(train_loss, test_loss))

    train_acc = fast_basic_net.accuracy(train_images, train_labels)
    test_acc = fast_basic_net.accuracy(test_images, test_labels)
    print("[Accuracy] train: {}, test: {}".format(train_acc, test_acc))

    log['accuracy_train'] = train_acc
    log['accuracy_test'] = test_acc

    pickle.dump(log, open(path.join(path.dirname(__file__ ), 'better_seven_layer_log.pkl'), "wb"))
