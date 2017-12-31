import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(X):
    if X.ndim == 1:
        exp = np.exp(X)
        return exp / np.sum(exp)
    elif X.ndim == 2:
        exp = np.exp(X)
        sum = np.sum(exp, axis=1, keepdims=True)
        return exp / sum

def cross_entropy(Y, T):
    return np.mean(np.sum(-np.log(Y) * T, axis=1))


if __name__ == '__main__':
    # print(sigmoid(3))
    # print(sigmoid(0))
    # print(sigmoid(-2))
    # print(sigmoid(-30))
    # print(sigmoid(np.array([-3, -0.1, 0, 0.8, 4])))

    one_dim = np.array([-0.5, 0.1, 0.3, 1])
    two_dim = np.array([
        [-0.5, 0.1, 0.3, 1],
        [3.0, 8.0, 0.4, 3.0]
    ])
    # print(softmax(one_dim))
    # print(softmax(two_dim))

    logits = softmax(two_dim)
    labels = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    print(cross_entropy(logits, labels))
