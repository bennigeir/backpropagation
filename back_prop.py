import numpy as np

from typing import Union
from util import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    return 0.0 if x <= -100 else np.divide(1, (1 + np.exp(-x)))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x) * (1 - sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    return np.sum(x * w), sigmoid(np.sum(x * w))


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    
    z0 = np.append(1.0, x)
    a1, z1, a2, y = [], [1.0], [], []

    for i in range(M):
        a1.append(perceptron(z0, W1[:,i])[0])
        z1.append(perceptron(z0, W1[:,i])[1])
    
    for i in range(K):
        a2.append(perceptron(z1, W2[:,i])[0])
        y.append(perceptron(z1, W2[:,i])[1])
    
    return y, z0, z1, a1, a2


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    
    dk = y - target_y
    dj = []

    for i in range(len(a1)):
        dj.append( d_sigmoid(a1[i]) * (np.sum(W2[i+1] * dk)))
    
    dE1, dE2 = np.zeros(W1.shape), np.zeros(W2.shape)
    
    for j in range(len(dj)):
        for i in range(len(z0)):
            dE1[i][j] = dj[j] * z0[i]
    
    for k in range(len(dk)):
        for j in range(len(z1)):
            dE2[j][k] = dk[k] * z1[j]
    
    return y, dE1, dE2
    

def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    guesses = [0] * len(X_train)
    N = len(X_train)
    misclassification_rate, Etotal = [], []
    
    for i in range(iterations):
        dE1_total, dE2_total = np.zeros(W1.shape), np.zeros(W2.shape)    

        err, misclass = 0, 0

        for j in range(N):
            target_y = np.zeros(K)
            target_y[t_train[j]] = 1.0
            
            y, dE1, dE2 = backprop(X_train[j], target_y, M, K, W1, W2)
            
            dE1_total += dE1
            dE2_total += dE2

            guesses[j] = np.argmax(y)

            err += ((target_y * np.log(np.array(y)))
                    + ((1 - target_y) * np.log( 1 - np.array(y))))
            
            if np.argmax(target_y) != guesses[j]:
                misclass += 1
        
        W1 -= eta * dE1_total / N
        W2 -= eta * dE2_total / N

        Etotal.append(np.sum(-err) / N)
        misclassification_rate.append(misclass / N)
    
    return W1, W2, Etotal, misclassification_rate, guesses


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    guesses = []
    for x in X:
        y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
        guesses.append(np.argmax(y))
    return guesses


def confusion_matrix(y_true, y_pred, classes):
    n = len(classes)
    
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(len(y_true)):
        x = y_true[i]
        y = y_pred[i]
        matrix[x][y] += 1
    
    return matrix