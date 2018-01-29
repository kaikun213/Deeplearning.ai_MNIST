# Libraries
import numpy as np

# helper functions
# ----------------------------------
# sigmoid function
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

#sigmoid derivative function
def sigmoid_derivative(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

# ReLu function
def relu(Z):
    return np.maximum(Z, 0)

# ReLu derivative
def relu_derivative(Z):
    return np.greater(Z, 0).astype(int)

# tanh function
def tanh(Z):
    return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))

# tanh derivative
def tanh_derivative(Z):
    return 1 - np.power(tanh(Z),2)

# ----------------------------------

# initialize parameters
def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)   # initialize random generator
    parameters = {}
    L = len(layer_dims)

    # initialize paramerters of each layer
    for l in range(1,L):
        parameters["W"+ str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters["b"+ str(l)] = np.zeros((layer_dims[l],1))

    # Assert correct Dimensionality
    assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


# feed forward propagation
# 1.) linear_forward calculating Z
# 2.) linear_activation_forward calculating A dependent on "relu"/"softmax", Z as input
# 3.) forward_propagation iterating over L-1 layers for relu and softmax in layer L

def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A_prev) + b

    # Assert correct Dimensionality
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (A_prev, W, b)

    return Z, cache

def relu_activation(Z):
    """
    ReLu activation function calculating Regression (Rectified Linear Unit)

    Arguments:
    Z -- pre-activation parameter calculated from WX+b

    Returns:
    A -- the output of the activation function
    activation_cache -- a python dictionary containing "A", stored for computing backward propagation
    """
    A = relu(Z)

    # Assert correct Dimensionality
    assert(A.shape == (Z.shape[0], Z.shape[1]))
    activation_cache = (Z)

    return A, activation_cache

def softmax_activation(Z):
    """
    Softmax activation function for multiclass classification

    Arguments:
    Z -- pre-activation parameter (n_y, 1)

    Returns:
    A -- the output of the actication function. Dimensionality is (n_y,1)
    activation_cache -- a python dictionary containing "Z", stored for computing backward propagation
    """

    t = np.exp(Z)
    A = t / np.sum(t, axis = 1, keepdims = True)

    # Assert correct Dimensionality (n_y, 1)
    assert(A.shape == (Z.shape[0], Z.shape[1]))
    activation_cache = (Z)

    return A, activation_cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu_activation(Z)

    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax_activation(Z)


    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->Softmax computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_softmax_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X                                     # initialize A0
    L = len(parameters) // 2                  # number of layers in the neural network = 3 (W and b thus divivded by 2)

    # Iterate [LINEAR -> RELU] for (L-1) layers
    for l in range(1, L):   # iterates 1,2 -> excludes upper limit L(3)
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    # LINEAR -> SOFTMAX for OUTPUT-LAYER
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    caches.append(cache)

    # 10 corresponds to the number of ouput classifications, 1 in binary-classification, 10 for character classification
    assert(AL.shape == (10,X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """
    Compute cost J for softmax (multiclass classification)

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (n_y, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (n_y, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    n_y = Y.shape[0]
    m = Y.shape[1]

    # Compute loss from AL and Y.
    # 1.) Compute Loss(vector) = negative of Y_j*log(Yhat_j) for each row j..n_y
    # 2.) Take the sum of all values => only where true value Y is one counts and as bigger it is as more Loss
    # 3.) Take average over all m examples
    cost = - 1/m * np.sum(- np.sum(Y*np.log(AL), axis = 0, keepdims = True), axis = 1) # Do not keep dimension => cost should become integer

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost


# backward propagation (using cached Z, W, b)
# 1.) linear_backward -- calculate dW, db
# 2.) linear_activation_backward dA dependent on "relu"/"softmax"
# 3.) backward_propagation iterating over all layers to calculate derivatives

def linear_backward(dZ, cache):
    """
    The linear portion of backward propagation for a single layer (layer l)

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

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    """
    The derivation of the ReLu activation function given previous dA & Z (chain rule)

    Arguments:
    dA -- post-activation gradient for current layer l
    activation_cache -- contains Z values for layers

    Returns:
    dZ -- Derivative of dJ in respect to dZ
    """
    Z = activation_cache[0]
    dZ = relu_derivative(Z) * dA

    return dZ


def softmax_backward(dA, activation_cache):
    """
    The derivation of the Softmax activation function given previous dA & Z (chain rule)

    Arguments:
    dA -- post-activation gradient for current layer l
    activation_cache -- contains Z values from the forward propagation in the current layer

    Returns:
    dZ -- Derivative of dJ in respect to dZ
    """

    # dA * g`(Z) = dA * (A * (1-A)) for sigmoid/softmax
    Z = activation_cache[0]
    dZ = sigmoid_derivative(Z) * dA

    return dZ



def linear_activation_backward(dA, cache, activation):
    """
    The backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    """
    The backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SOFTMAX group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

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
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost (-y*log(a)-(1-y)*log(1-a)) with respect to AL=a

    # Assert softmax calculation correct
    _, activation_cache = caches[L-1]

    # DEBUG
    # print("AL", AL)
    # print("dAL", dAL)
    # print("dZ", sigmoid_derivative(activation_cache[0]))
    # print("dAL*dZ", dAL*sigmoid_derivative(activation_cache[0]))
    # print("Y", Y)
    # print("Softmax-comp", softmax_backward(dAL, activation_cache))
    # print("AL-Y", AL-Y)
    # assert(softmax_backward(dAL, activation_cache) == (AL-Y)).all()

    # Lth layer (SOFTMAX -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    # current_cache = caches[L-1]
    # grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "softmax")

    # Alternative to initialize dAL
    grads["dA" + str(L)] = AL - Y


    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]

        # DEBUG
        # print("l", l)
        # print("caches length", len(caches))
        # print("dZ", grads["dA"+str(l+2)].shape)
        # linear_cache, activation_cache = current_cache
        # print("W", linear_cache[1].shape)

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)], current_cache, "relu")
        grads["dA" + str(l+1)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


# UPDATE Parameters
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

    # Update rule for each parameter
    for l in range(1,L):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters

# L-Layer Deep Neural network
def L_layer_model(X, Y, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    A L-layer deep neural network: [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX.

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
    costs = []                         # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters(layer_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX.
        AL, caches = forward_propagation(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = backward_propagation(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters
