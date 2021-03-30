import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Iris.csv')
X = dataset[["SepalLengthCm", "SepalWidthCm",
             "PetalLengthCm", "PetalWidthCm"]].values
Y = dataset["Species"]
Y = Y.replace({'Iris-setosa': 0, 'Iris-versicolor': 0.5,
               'Iris-virginica': 1}).values


class NeuralNetwork(object):
    def __init__(self):
        self.learning_rate = 0.1
        self.output_size = 3
        self.hidden_size = 2
        self.weights_ih = np.random.randn(2, 4)
        self.weights_ho = np.random.randn(3, 2)
        self.bias_h = np.random.randn(2)*-1
        self.bias_o = np.random.randn(3)*-1

    # ustawia wynik layer_three_y, czyli macierz trzyelementową
    def feed_forward(self, input):
        self.layer_two_y = sigmoid_unipolar((np.matmul(
            self.weights_ih, input)) + self.bias_h)                  # layer_two_y dim: (2x1)
        self.layer_three_y = sigmoid_unipolar((np.matmul(
            self.weights_ho, self.layer_two_y)) + self.bias_o)     # layer_three_y dim: (3x1)
        return self.layer_three_y

    def propagate_backwards(self, input, guess, output):
        self.output_errors = self.target - self.layer_three_y
        self.hidden_errors = np.matmul(self.weights_ho.T, output_errors)

        delta_weights_ho = np.matmul(
            self.learning_rate*self.output_errors*sigmoid_unipolar(layer_three_y, True), self.layer_two_y.T)
        delta_weights_ih = np.matmul(
            self.learning_Rate*self.hidden_errors*sigmoid_unipolar(two, True), self.input.T)

    # x - input, y - output (guess),
    def train(self, X, y):
        output = self.feed_forward(X)
        self.propagate_backwards(X, layer_three_y)


def sigmoid_unipolar(s, get_derivative=False):
    if (get_derivative):
        return s*(1-s)
    return 1/(1+np.exp(-s))


def sigmoid_bipolar(s, get_derivative=False):
    if (get_derivative):
        return s*(1-s)
    return (2/(1+np.exp(-s))-1)


nn = NeuralNetwork()

print(np.array([0, .5, 1]))
