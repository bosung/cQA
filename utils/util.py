import numpy as np


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)