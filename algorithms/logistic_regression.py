"""
Logistic regression model
"""

import numpy as np
import math


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5  # To threshold the sigmoid
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-z))

    def sigmoid_single(self, z: float) -> float:
        return 1 / (1 + math.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        for _ in range(self.epochs):
            for j, W in enumerate(self.w):
                grad_sum = np.zeros(D)
                yi = 1
                for i, xi in enumerate(X_train):
                    label = y_train[i]
                    if label == j:
                        yi = 1
                    else:
                        yi = -1
                    y_pred = np.dot(W, xi)
                    gradient = self.sigmoid_single(-yi * y_pred) * yi * xi
                    grad_sum += gradient
                grad_sum /= N
                self.w[j] -= self.lr * (self.weight_decay * W - grad_sum)


                # input = -yi * np.dot(X_train, W)
                # sum = np.sum((self.sigmoid(input) * yi)[:, np.newaxis] * X_train, axis=0)
                # self.w[j] = W - self.lr * (self.weight_decay * W - (1/N) * sum)
        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N, D = X_test.shape
        result = np.zeros(N)
        for i, image in enumerate(X_test):
            output = np.zeros(D)
            for j, W in enumerate(self.w):
                output[j] = np.dot(W, image)
            result[i] = np.argmax(output)
        return result
