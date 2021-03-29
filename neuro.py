import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import training dataset
dataset = pd.read_csv('Iris.csv')
train_x = dataset.iloc[:, 1:5].values
train_y = dataset.iloc[:, 5].values
print(train_y)


class NeuralNetwork(object):
    def __init__(self, input, weights, hiddenLayerSize):
        self.input = input
        self.weights = np.random.rand(input.size, hiddenLayerSize)
        self.bias = np.random.rand(input.size)*(-1)
        self.hiddenLayerSize = hiddenLayerSize

    def feed_forward(self):
        weightedSum = np.matmul(self.weights, input)
        weightedSum += bias
        guess = 1/(1+np.exp(-weightedSum))
        return guess


def test_network():
    neural_layer = NeuralNetwork(n)
