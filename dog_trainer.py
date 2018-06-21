# Arjun Subramonian

# One vs. All Multi-Class Classification (not AVA)
# Simple Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

# - Build the general architecture of a learning algorithm, including:
#     - Initialize the parameters of the model
#     - Calculating the cost function and its gradient
#     - Using an optimization algorithm (gradient descent)
#     - Learn the parameters for the model by minimizing the cost  
#     - Use the learned parameters to make predictions
#     - Analyze the results and conclude
# - Gather all three functions above into a main model function, in the right order.

# The main steps for building a Neural Network are:
# 1. Define the model structure (such as number of input features) 
# 2. Initialize the model's parameters
# 3. Loop:
#     - Calculate current loss (forward propagation)
#     - Calculate current gradient (backward propagation)
#     - Update parameters (gradient descent)

# Helper functions

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    # This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    # Argument:
    # dim -- size of the w vector we want (or number of parameters in this case)
    
    # Returns:
    # w -- initialized vector of shape (dim, 1)
    # b -- initialized scalar (corresponds to the bias)
    
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def propagate(w, b, X, Y):
    # Implement the cost function and its gradient for propagation

    # Arguments:
    # w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    # b -- bias, a scalar
    # X -- data of size (num_px * num_px * 3, number of examples)
    # Y -- true "label" vector (containing 0 if non-breed, 1 if breed) of size (1, number of examples)

    # Return:
    # cost -- negative log-likelihood cost for logistic regression
    # dw -- gradient of the loss with respect to w, thus same shape as w
    # db -- gradient of the loss with respect to b, thus same shape as b
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)                                                 # compute activation
    cost = (-1 / m) * (np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T))   # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1 / m) * (np.dot(X, (A - Y).T))
    db = (1 / m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    # This function optimizes w and b by running a gradient descent algorithm
    
    # Arguments:
    # w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    # b -- bias, a scalar
    # X -- data of shape (num_px * num_px * 3, number of examples)
    # Y -- true "label" vector (containing 0 if non-breed, 1 if breed) of size (1, number of examples)
    # num_iterations -- number of iterations of the optimization loop
    # learning_rate -- learning rate of the gradient descent update rule
    # print_cost -- True to print the loss every 100 steps
    
    # Returns:
    # params -- dictionary containing the weights w and bias b
    # grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    # costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # Perform updates
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    # Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    # Arguments:
    # w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    # b -- bias, a scalar
    # X -- data of size (num_px * num_px * 3, number of examples)
    
    # Returns:
    # A -- a numpy array (vector) containing all probabilities of a breed being present in the picture
    
    w = w.reshape(X.shape[0], 1)
    
    # Compute A, which predicts the probabilities of a breed being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    return A

### ONLY USE IF HAVE TEST SET!!!
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # Builds the logistic regression model by calling the function you've implemented previously
    
    # Arguments:
    # X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    # Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    # X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    # Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    # num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    # learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    # print_cost -- Set to true to print the cost every 100 iterations
    
    # Returns:
    # d -- dictionary containing information about the model.
    
    # Initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
            "costs": costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations
        }
    
    return d

num_px = 128

'''Make a dictionary with the breeds as keys.'''
breeds = {}
with open('breeds.txt') as file:
    for line in file:
        breeds[line.strip()] = []

for breed in breeds:
    print(breed)

    X = np.loadtxt('./data/' + breed + '_X.txt')
    # CAUTION - this returns a rank one array, not a row vector
    Y = np.loadtxt('./data/' + breed + '_Y.txt')

    col = Y.shape[0]

    X = np.hstack((X, np.zeros((num_px * num_px * 3, 3 * (len(breeds) - 1)))))
    Y = np.hstack((Y, np.zeros(3 * (len(breeds) - 1))))

    for other in breeds:
        if other == breed:
            continue
        else:
            X_other = np.loadtxt('./data/' + other + '_X.txt', usecols = (0, 1, 2))

            # Make first three images in each other class negative training data for current breed.
            X[:, col:(col + 3)] = X_other

            col += 3

    # Scale RGB values to [0, 1] using mini and maxi.
    mini = np.amin(X)
    maxi = np.amax(X)
    X = (X - mini) / (maxi - mini)

    # Initialize parameters with zeros
    w, b = initialize_with_zeros(X.shape[0])

    # Gradient descent
    num_iterations = 2000
    # 0.005
    learning_rate = 0.01
    print_cost = True
    parameters, grads, costs = optimize(w, b, X, Y, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary
    w = parameters["w"]
    b = parameters["b"]

    np.savetxt('./trained_values/' + breed + '_w.txt', w)
    with open('./trained_values/' + breed + '_b.txt', 'w') as f:
        f.write(str(b))
    with open('./trained_values/' + breed + '_mini.txt', 'w') as f:
        f.write(str(mini))
    with open('./trained_values/' + breed + '_maxi.txt', 'w') as f:
        f.write(str(maxi))
