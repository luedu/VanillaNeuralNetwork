import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    """
    Sigmoid activation
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def sigmoid_backward(dA, cache):
    """
    Backward propagation for a single SIGMOID unit

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def softmax(Z):
    """
    Softmax activation
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of softmax(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = np.exp(Z)    
    A = A/np.sum(A, axis = 0, keepdims = 1)
    cache = Z
    
    return A, cache

def softmax_backward(dA, cache):
    """
    Backward propagation for a single SOFTMAX unit

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache

    s = np.exp(Z)
    s = s / np.sum(s, axis = 0, keepdims = 1)
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ

def initialize_parameters_deep(layer_dims, method):
    """
    Parameter initialization
        
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    method -- "random" or "He"
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        if method == "random":
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])
        else:
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

    
def linear_activation_forward(A_prev, W, b, activation, keep_prob):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "softmax"
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache", "activation_cache" and "D";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    
    D = None
    if keep_prob < 1:                                       # Dropout
            D = np.random.rand(A.shape[0], A.shape[1])
            D = D < keep_prob                               # Thresholding
            A = A * D                                       # Node shutdown
            A = A / keep_prob                               # Scale the value of neurons that haven't been shut down

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache, D)

    return A, cache

def L_model_forward(X, parameters, keep_prob):
    """
    Implement forward propagation for the [LINEAR->SIGMOID]*(L-1)->LINEAR->SOFTMAX computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_sigmoid_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_softmax_forward() (there is one, indexed L-1)
                D, the dropout mask
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> SIGMOID]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "sigmoid", keep_prob = keep_prob)
        caches.append(cache)
    
    # Implement LINEAR -> SOFTMAX. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax", keep_prob = 1)
    caches.append(cache)
    
    assert(AL.shape == (parameters['W' + str(L)].shape[0],X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function as the cross-entropy

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (number of classes, number of examples)
    Y -- true "label" vector, shape (number of classes, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y
    cost = (-1 / m) * np.sum(np.sum(np.multiply(Y, np.log(AL + 1e-10)) + np.multiply(1 - Y, np.log(1 - AL + 1e-10)), axis = 1))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, cache[0].T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation, keep_prob):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache, D) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "softmax"
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache, D = cache

    if keep_prob < 1:                           # Dropout
        dA = dA * D                             # Apply mask
        dA = dA / keep_prob                     # Scale the value of neurons
    
    if activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)    
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, keep_prob):
    """
    Implement the backward propagation for the [LINEAR->SIGMOID] * (L-1) -> LINEAR -> SOFTMAX group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing a one-hot column vector indicating the class)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "sigmoid" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "softmax" (there is one, index L-1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> SOFTMAX) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "softmax", keep_prob = 1)
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "sigmoid", keep_prob = keep_prob)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters

def L_layer_model(X, Y, layers_dims, init_method="random", learning_rate=0.0075, num_iterations=3000, print_cost=False, keep_prob = 1): #lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->SIGMOID]*(L-1)->LINEAR->SOFTMAX
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    init_method -- initialization method: "random" or "He"
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims, init_method)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> SIGMOID]*(L-1) -> LINEAR -> SOFTMAX
        AL, caches = L_model_forward(X, parameters, keep_prob)
        
        # Compute cost
        cost = compute_cost(AL, Y)
    
        # Backward propagation
        grads = L_model_backward(AL, Y, caches, keep_prob)
 
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X, parameters):
    """
    Predict the output of the NN
    """

    AL, cache = L_model_forward(X, parameters, 1)

    p = None
        
    p = np.zeros(AL.shape)

    # Get the indexes of the max values in all columns
    tmp_prd_idx = np.argmax(AL, axis = 0)
    p[tmp_prd_idx, np.arange(len(tmp_prd_idx))] = 1

    return p

def predict_accuracy(X, Y, parameters):
    """
    Predict the output of the NN and compute its accuracy
    """

    m = X.shape[1]
    p = predict(X, parameters)
        
    tmp_p = p.astype(bool)
    tmp_y = Y.astype(bool)
    print("Accuracy: " + str(np.sum((tmp_p & tmp_y)/m)))
    
    return p
