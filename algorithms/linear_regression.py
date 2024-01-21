"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        for _ in range(self.epochs):
            for j, W in enumerate(self.w):
                sum = np.zeros(D)
                for i, xi, in enumerate(X_train):
                    label = y_train[i]
                    yi = 1
                    if label == j:
                        yi = 1
                    else:
                        yi = -1
                    sum = sum + 2 * (np.dot(W, xi) - yi) * xi
                self.w[j] = W - self.lr * (self.weight_decay  * W + (1//N) * sum)
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
