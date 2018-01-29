# My libraries
import numpy as np

import mnist_loader
import network

def one_hot(Y, C):
    """
    One hot encoding for input values Y which range from 0..C-1 classes
    E.g. [0 5 9 2 4] will be converted to a matrix with a one in the respective class row and each number as own column-vector

    Arguments:
    Y -- Numpy array of labels (labeled with classes-number)
    C -- Range of classes

    Returns:
    one_hot -- converted matrix
    """
    # dimensions: (# classes, # of items)
    identity_matrix = np.eye(C) # creates Identity matrix with dimension C
    one_hot = identity_matrix[Y] # Select the column from identity_matrix that correspond to the indices in Y

    return one_hot

def deep_net_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()

    X = training_data[0]
    Y = training_data[1]

    # one_hot_encoding and reshape both (Transpose - flip rows/columns, so m=50.000=columns)
    X_reshaped = X.reshape(784, 50000)
    Y_reshaped = one_hot(Y, 10).reshape(10, 50000)

    # train
    layer_dims = [784, 20, 25, 15, 10] #  5-layer model
    parameters = network.L_layer_model(X_reshaped, Y_reshaped, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False)

    # test
    #predictions = [int(a) for a in clf.predict(test_data[0])]
    #num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    #print "Baseline classifier using an SVM."
    #print "%s of %s values correct." % (num_correct, len(test_data[1]))


if __name__ == "__main__":
    deep_net_baseline()
