import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))







if __name__ == '__main__':
    print(sigmoid(3))
    print(sigmoid(0))
    print(sigmoid(-2))
    print(sigmoid(-30))
    print(sigmoid(np.array([-3, -0.1, 0, 0.8, 4])))
