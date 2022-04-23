"""Neural network model."""

from typing import Sequence
from unittest import result

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.
    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
            self,
            input_size: int,
            hidden_sizes: Sequence[int],
            output_size: int,
            num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.v = None
        self.m = None
        self.epsilon = None
        self.beta_2 = None
        self.beta_1 = None
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return X @ W + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return X > 0

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me

        X = X - np.max(X, axis=1).reshape(X.shape[0], -1)
        X = np.exp(X)
        return X / np.sum(X, axis=1).reshape(X.shape[0], -1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.
        self.outputs[0] = X
        for i in range(1, self.num_layers):
            self.outputs[i] = self.linear(self.params['W' + str(i)], self.outputs[i - 1], self.params['b' + str(i)])
            self.outputs[i] = self.relu(self.outputs[i])
        self.outputs[self.num_layers] = self.softmax(
            self.linear(self.params['W' + str(self.num_layers)], self.outputs[self.num_layers - 1],
                        self.params['b' + str(self.num_layers)]))
        return self.outputs[self.num_layers]

    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Note: both gradients and loss should include regularization.
        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        m = y.shape[0]
        p = np.copy(self.outputs[self.num_layers])
        loss = np.sum(-np.log(p[range(m), y])) / m
        for i in range(1, self.num_layers + 1):
            loss += np.sum(self.params["W" + str(i)] * self.params["W" + str(i)]) * reg
        p[range(m), y] -= 1
        grad = p / m

        self.gradients['L' + str(self.num_layers)] = grad

        for i in reversed(range(1, self.num_layers + 1)):
            L_grad = self.relu_grad(self.outputs[i]) * self.gradients['L' + str(i)]
            self.gradients['L' + str(i - 1)] = L_grad @ np.transpose(self.params['W' + str(i)])
            self.gradients['W' + str(i)] = 2 * reg * self.params['W' + str(i)] + \
                                           np.transpose(self.outputs[i - 1]) @ L_grad
            self.gradients['b' + str(i)] = np.sum(L_grad, axis=0)

        return loss

    def update(
            self,
            lr: float = 0.001,
            b1: float = 0.9,
            b2: float = 0.999,
            eps: float = 1e-8,
            opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "ADAM":
            self.beta_1 = b1
            self.beta_2 = b2
            self.epsilon = eps
            self.m = {}
            self.v = {}

            for i in range(1, self.num_layers + 1):
                self.m["m" + str(i)] = np.zeros_like(self.params['W' + str(i)])
                self.v["v" + str(i)] = np.zeros_like(self.params['W' + str(i)])

            for i in range(1, self.num_layers + 1):
                self.m["m" + str(i)] = b1 * self.m["m" + str(i)] + (1.0 - b1) * self.gradients['W' + str(i)]
                self.v["v" + str(i)] = b2 * self.v["v" + str(i)] + (1.0 - b2) * (self.gradients['W' + str(i)] ** 2)

                new_grad = self.m["m" + str(i)] * lr / (np.sqrt(self.v["v" + str(i)]) + eps)
                self.params['W' + str(i)] -= lr * new_grad
                self.params['b' + str(i)] -= lr * self.gradients['b' + str(i)]
        else:
            for i in range(1, self.num_layers + 1):
                self.params['W' + str(i)] -= lr * self.gradients['W' + str(i)]
                self.params['b' + str(i)] -= lr * self.gradients['b' + str(i)]
        pass
