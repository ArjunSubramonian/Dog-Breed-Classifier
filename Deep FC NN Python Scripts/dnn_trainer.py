import numpy as np
from dnn_helper import *

# Build two different models:
# - A 2-layer neural network
# - An L-layer deep neural network

# Follow the Deep Learning methodology to build the model:
#     1. Initialize parameters / Define hyperparameters
#     2. Loop for num_iterations:
#         a. Forward propagation
#         b. Compute cost function
#         c. Backward propagation
#         d. Update parameters (using parameters, and grads from backprop) 
#     4. Use trained parameters to predict labels

# Two-layer neural network

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
    
    return parameters

# L-layer Neural Network

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    return parameters

### CONSTANTS DEFINING THE MODEL ####
num_px = 128

# n_x = num_px * num_px * 3
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)

L = 4
layers_dims = [num_px * num_px * 3, 20, 7, 5, 1] #  L-layer model

'''Make a dictionary with the breeds as keys.'''
breeds = {}
with open('../Classes/breeds.txt') as file:
    for line in file:
        breeds[line.strip()] = []

for breed in breeds:
    print(breed)

    X = np.loadtxt('../data/' + breed + '_X.txt')
    # CAUTION - this returns a rank one array, not a row vector
    Y = np.loadtxt('../data/' + breed + '_Y.txt')

    col = Y.shape[0]

    append = int(col // (len(breeds) - 1))

    X = np.hstack((X, np.zeros((num_px * num_px * 3, append * (len(breeds) - 1)))))
    Y = np.hstack((Y, np.zeros(append * (len(breeds) - 1))))

    use = []
    for c in range(0, append):
        use.append(c)
    use = tuple(use)

    for other in breeds:
        if other == breed:
            continue
        else:
            X_other = np.loadtxt('../data/' + other + '_X.txt', usecols = use)

            # Make first n images in each other class negative training data for current breed.
            X[:, col:(col + append)] = X_other

            col += append

    # Scale RGB values to approximately [0, 1].
    X /= 255

    # parameters = two_layer_model(X, Y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
    # Retrieve W1, b1, W2, b2 from parameters
    # W1 = parameters["W1"]
    # b1 = parameters["b1"]
    # W2 = parameters["W2"]
    # b2 = parameters["b2"]

    parameters = L_layer_model(X, Y, layers_dims, learning_rate = 0.005, num_iterations = 2500, print_cost = True)
    for l in range(1, L + 1):
        w = parameters["W" + str(l)]
        b = parameters["b" + str(l)]

        np.savetxt('../trained_values/' + breed + '_W' + str(l) + '.txt', w)
        np.savetxt('../trained_values/' + breed + '_b' + str(l) + '.txt', b)
   
    # with open('../trained_values/' + breed + '_mini.txt', 'w') as f:
    #     f.write(str(mini))
    # with open('../trained_values/' + breed + '_maxi.txt', 'w') as f:
    #     f.write(str(maxi))
