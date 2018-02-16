import sys, os.path as path
sys.path.append(path.abspath(path.join(__file__ ,"../../../..")))
import numpy as np
import pickle
from lib.MNIST import MNIST
from lib.layers import DenseLayer, BatchNormLayer, ReluLayer, SoftmaxCrossEntropyLayer, SigmoidLayer
from collections import OrderedDict

class BatchNormNet:
    def __init__(self, weight_stddev=0.01, uses_batch_norm=False, experiment_num=0):
        self.params = None
        self.layers = None
        
        self.weight_stddev = weight_stddev
        self.uses_batch_norm = uses_batch_norm
        self.experiment_num = experiment_num

        self.init_params()
        self.init_layers()

    def init_params(self):
        self.params = {}
        self.params['W1'] = np.random.randn(28*28, 64) * self.weight_stddev
        self.params['b1'] = np.zeros(64)
        if self.uses_batch_norm:
            self.params['gamma1'] = np.ones(64)
            self.params['beta1'] = np.zeros(64)

        self.params['W2'] = np.random.randn(64, 50) * self.weight_stddev
        self.params['b2'] = np.zeros(50)
        if self.uses_batch_norm:
            self.params['gamma2'] = np.ones(50)
            self.params['beta2'] = np.zeros(50)

        self.params['W3'] = np.random.randn(50, 40) * self.weight_stddev
        self.params['b3'] = np.zeros(40)
        if self.uses_batch_norm:
            self.params['gamma3'] = np.ones(40)
            self.params['beta3'] = np.zeros(40)
        
        self.params['W4'] = np.random.randn(40, 30) * self.weight_stddev
        self.params['b4'] = np.zeros(30)
        if self.uses_batch_norm:
            self.params['gamma4'] = np.ones(30)
            self.params['beta4'] = np.zeros(30)
        
        self.params['W5'] = np.random.randn(30, 30) * self.weight_stddev
        self.params['b5'] = np.zeros(30)
        if self.uses_batch_norm:
            self.params['gamma5'] = np.ones(30)
            self.params['beta5'] = np.zeros(30)
        
        self.params['W6'] = np.random.randn(30, 30) * self.weight_stddev
        self.params['b6'] = np.zeros(30)
        if self.uses_batch_norm:
            self.params['gamma6'] = np.ones(30)
            self.params['beta6'] = np.zeros(30)
        
        self.params['W7'] = np.random.randn(30, 10) * self.weight_stddev
        self.params['b7'] = np.zeros(10)

    def init_layers(self):
        self.layers = OrderedDict()
        self.layers['Dense1'] = DenseLayer(self.params['W1'], self.params['b1'])
        if self.uses_batch_norm:
            self.layers['BatchNorm1'] = BatchNormLayer(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = ReluLayer()

        self.layers['Dense2'] = DenseLayer(self.params['W2'], self.params['b2'])
        if self.uses_batch_norm:
            self.layers['BatchNorm2'] = BatchNormLayer(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = ReluLayer()

        self.layers['Dense3'] = DenseLayer(self.params['W3'], self.params['b3'])
        if self.uses_batch_norm:
            self.layers['BatchNorm3'] = BatchNormLayer(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = ReluLayer()
        
        self.layers['Dense4'] = DenseLayer(self.params['W4'], self.params['b4'])
        if self.uses_batch_norm:
            self.layers['BatchNorm4'] = BatchNormLayer(self.params['gamma4'], self.params['beta4'])
        self.layers['Relu4'] = ReluLayer()
        
        self.layers['Dense5'] = DenseLayer(self.params['W5'], self.params['b5'])
        if self.uses_batch_norm:
            self.layers['BatchNorm5'] = BatchNormLayer(self.params['gamma5'], self.params['beta5'])
        self.layers['Relu5'] = ReluLayer()
        
        self.layers['Dense6'] = DenseLayer(self.params['W6'], self.params['b6'])
        if self.uses_batch_norm:
            self.layers['BatchNorm6'] = BatchNormLayer(self.params['gamma6'], self.params['beta6'])
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
        ETA = 0.1
        grads = self.gradients(X, T)
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

        if self.uses_batch_norm:
            gradients['gamma1'] = self.layers['BatchNorm1'].dgamma
            gradients['beta1'] = self.layers['BatchNorm1'].dbeta
            gradients['gamma2'] = self.layers['BatchNorm2'].dgamma
            gradients['beta2'] = self.layers['BatchNorm2'].dbeta
            gradients['gamma3'] = self.layers['BatchNorm3'].dgamma
            gradients['beta3'] = self.layers['BatchNorm3'].dbeta
            gradients['gamma4'] = self.layers['BatchNorm4'].dgamma
            gradients['beta4'] = self.layers['BatchNorm4'].dbeta
            gradients['gamma5'] = self.layers['BatchNorm5'].dgamma
            gradients['beta5'] = self.layers['BatchNorm5'].dbeta
            gradients['gamma6'] = self.layers['BatchNorm6'].dgamma
            gradients['beta6'] = self.layers['BatchNorm6'].dbeta

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

def experiment(weight_stddev, uses_batch_norm, experiment_num):
    np.random.seed(1229)
    batch_norm_net = BatchNormNet(weight_stddev, uses_batch_norm, experiment_num)

    mnist = MNIST()
    train_images, train_labels, test_images, test_labels = mnist.get_dataset()

    log = {
        'accuracy_train': [],
        'accuracy_train_itr': [],
        'accuracy_test': [],
        'accuracy_test_itr': [],
        'weight_stddev': weight_stddev,
        'uses_batch_norm': uses_batch_norm
    }

    epochs = 5
    train_size = train_images.shape[0]
    batch_size = 100
    iteration_per_epoch = train_size // batch_size
    total_iterations = iteration_per_epoch * epochs

    itr = 0
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))

        batch_norm_net.is_training = True
        for _ in range(iteration_per_epoch):

            if itr % 60 == 0:
                batch_norm_net.is_training = False
                train_acc = batch_norm_net.accuracy(train_images, train_labels)
                test_acc = batch_norm_net.accuracy(test_images, test_labels)
                log['accuracy_train'].append(train_acc)
                log['accuracy_train_itr'].append(itr)
                log['accuracy_test'].append(test_acc)
                log['accuracy_test_itr'].append(itr)
                print("[Accuracy] train: {}, test: {}".format(train_acc, test_acc))
                batch_norm_net.is_training = True

            batch_mask = np.random.choice(train_size, batch_size)
            batch_images = train_images[batch_mask]
            batch_labels = train_labels[batch_mask]

            batch_norm_net.gradient_descent(batch_images, batch_labels)
            itr += 1

        batch_norm_net.is_training = False

    print("Done!")

    train_acc = batch_norm_net.accuracy(train_images, train_labels)
    test_acc = batch_norm_net.accuracy(test_images, test_labels)
    log['accuracy_train'].append(train_acc)
    log['accuracy_train_itr'].append(itr)
    log['accuracy_test'].append(test_acc)
    log['accuracy_test_itr'].append(itr)
    print("[Accuracy] train: {}, test: {}".format(train_acc, test_acc))

    pickle.dump(log, open(path.join(path.dirname(__file__ ), 'experiment_{}_{}.pkl'.format(experiment_num, uses_batch_norm)), "wb"))

if __name__ == '__main__':
    print("this is main")

    for i in range(2):
        r = -4 * np.random.rand()
        weight_stddev = 10 ** r
        print("======== stddev: ", weight_stddev)
        experiment(weight_stddev, True, i)
        experiment(weight_stddev, False, i)
