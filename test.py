import numpy as np
from dnn_utils import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Configuration
test_samples_proportion = 0.2       # 20%
layers_dims = [4, 10, 7, 3]
initialization_method = "He"
num_iter = 30000
learn_rate = 0.3
print_c = True
dropout_prob = 0.7                  # Only applicable to the hidden layers

# Import the dataset, standardize it and split it into training and test sets
X, y = load_iris(True)
X -= np.mean(X, axis=0)
X /= np.sum(np.std(X, axis=0), axis=0)
encoder = OneHotEncoder(sparse=False) # One-hot encoding
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = test_samples_proportion)
train_y = (encoder.fit_transform(train_y.reshape(-1, 1))).T
test_y = (encoder.fit_transform(test_y.reshape(-1, 1))).T
train_x = train_x.T
test_x = test_x.T

# Print dataset information
print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
print("train_y's shape: " + str(train_y.shape))
print("test_y's shape: " + str(test_y.shape))

# Build the network
parameters = L_layer_model(train_x, train_y, layers_dims, init_method = initialization_method, learning_rate = learn_rate, num_iterations = num_iter, print_cost = print_c, keep_prob = dropout_prob)

# Evaluate performance
print("\n--- Training ---")
pred_train = predict_accuracy(train_x, train_y, parameters)
print("\n--- Test ---")
pred_test = predict_accuracy(test_x, test_y, parameters)