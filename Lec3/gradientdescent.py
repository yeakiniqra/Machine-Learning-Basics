# Gradient Descent Algorithm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Function to compute the gradient of the cost function
def compute_gradient(X, y, theta):
    m = len(y)
    gradient = (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
    return gradient

# Function to compute the cost function
def compute_cost(X, y, theta):
    m = len(y)
    cost = (1/(2*m)) * np.sum(np.square(np.dot(X, theta) - y))
    return cost

# Function to perform gradient descent
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        theta = theta - learning_rate * compute_gradient(X, y, theta)
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history

# Function to normalize the features
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std 

# Function to predict the output
def predict(X, mean, std, theta):
    X = (X - mean) / std
    y_pred = np.dot(X, theta)
    return y_pred 

# Load the data
data = pd.read_csv('test_scores.csv')
# Convert 'math' and 'cs' columns to float
data['math'] = data['math'].astype(float)
data['cs'] = data['cs'].astype(float)

X = data.iloc[:, 1].values  # 'math' scores
y = data.iloc[:, 2].values  # 'cs' scores




# Convert X to a numpy array of floats
X = np.array(X, dtype=float)

# Normalize the features
X, mean, std = normalize_features(X)

# Add a column of ones to X
X = np.c_[np.ones(X.shape[0]), X]

# Initialize theta
theta = np.zeros(X.shape[1])

# Set the hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Perform gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, num_iterations)

# Predict the output
y_pred = predict(X, mean, std, theta)

# Plot the data
plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
plt.plot(data.iloc[:, 0], y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()

# Plot the cost function
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.show()

# Print the final theta
print('Final theta:', theta)

# Print the cost
print('Final cost:', cost_history[-1])

# Print the predicted output
print('Predicted output:', y_pred)

# Print the actual output
print('Actual output:', y)

# Print the mean squared error
print('Mean squared error:', np.mean((y_pred - y) ** 2))

# Print the coefficient of determination
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print('Coefficient of determination:', r_squared)

# Print the root mean squared error
print('Root mean squared error:', np.sqrt(np.mean((y_pred - y) ** 2)))

# Print the mean absolute error
print('Mean absolute error:', np.mean(np.abs(y_pred - y)))

# Print the mean absolute percentage error
print('Mean absolute percentage error:', np.mean(np.abs((y - y_pred) / y) * 100))

# Print the mean squared logarithmic error
print('Mean squared logarithmic error:', np.mean((np.log(y + 1) - np.log(y_pred + 1)) ** 2))

# Print the median absolute error
print('Median absolute error:', np.median(np.abs(y_pred - y)))

# Print the explained variance score
print('Explained variance score:', 1 - (np.var(y - y_pred) / np.var(y)))

# Print the maximum error
print('Maximum error:', np.max(np.abs(y_pred - y)))

# Print the minimum error
print('Minimum error:', np.min(np.abs(y_pred - y)))

# Print the mean error
print('Mean error:', np.mean(np.abs(y_pred - y)))

# Print the median error
print('Median error:', np.median(np.abs(y_pred - y)))

# Print the standard deviation of the error
print('Standard deviation of the error:', np.std(np.abs(y_pred - y)))

# Print the variance of the error
print('Variance of the error:', np.var(np.abs(y_pred - y)))

# Print the skewness of the error
print('Skewness of the error:', pd.Series(y_pred - y).skew())

