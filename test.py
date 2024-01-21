# Prepare Packages
import numpy as np
import matplotlib.pyplot as plt

from utils.data_processing import get_cifar10_data

# Use a subset of CIFAR10 for the assignment
dataset = get_cifar10_data(
    subset_train=5000,
    subset_val=250,
    subset_test=500,
)
x_train = dataset["x_train"]
y_train = dataset["y_train"]
x_val = dataset["x_val"]
y_val = dataset["y_val"]
x_test = dataset["x_test"]
y_test = dataset["y_test"]


# Import the algorithm implementation (TODO: Complete the Linear Regression in algorithms/linear_regression.py)
from algorithms import Linear
from utils.evaluation import get_classification_accuracy

num_classes = 10  # Cifar10 dataset has 10 different classes

# Initialize hyper-parameters
learning_rate = 0.0001  # You will be later asked to experiment with different learning rates and report results
num_epochs_total = 200  # Total number of epochs to train the classifier
epochs_per_evaluation = 10  # Epochs per step of evaluation; We will evaluate our model regularly during training
N, D = dataset[
    "x_train"
].shape  # Get training data shape, N: Number of examples, D:Dimensionality of the data
weight_decay = 0.0

# Insert additional scalar term 1 in the samples to account for the bias as discussed in class
x_train = np.insert(x_train, D, values=1, axis=1)
x_val = np.insert(x_val, D, values=1, axis=1)
x_test = np.insert(x_test, D, values=1, axis=1)

# Create a linear regression object
linear_regression = Linear(
    num_classes, learning_rate, epochs_per_evaluation, weight_decay
)

# Randomly initialize the weights and biases
weights = np.random.randn(num_classes, D + 1) * 0.0001

weights = linear_regression.train(x_train, y_train, weights)
result = linear_regression.predict(x_train)
print("y train: ", y_train)
for i in range(10):
    print(i,": ", y_train[i])
print("result: ", result)
for i in range(10):
    print(i,": ", result[i])