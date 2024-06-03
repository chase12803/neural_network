import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf  # only using tf to access the MNIST database
import numpy as np
import matplotlib.pyplot as pyplot


def norm(X):
    """
    The values of each pixel in the MNIST dataset are between 0 and 255, we want to normalize this so it is a value
    from 0 to 1.
    :param X: The activation of a pixel of an image in the MNIST dataset
    :return: The normalized activation
    """
    return X / 255


#getting the data
(train_X, train_y), (dev_X, dev_y) = tf.keras.datasets.mnist.load_data()
t1, m1, n1 = train_X.shape
train_X1 = norm(np.reshape(train_X, (t1, m1 * n1)))
t2, m2, n2 = dev_X.shape
dev_X1 = norm(np.reshape(dev_X, (t2, m2 * n2)))


def test_plot():
    # printing the shapes of the vectors
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_dev:  ' + str(dev_X.shape))
    print('Y_dev:  ' + str(dev_y.shape))

    pyplot.imshow(train_X[2], cmap=pyplot.get_cmap('gray'))
    pyplot.show()


# Initialize all weights as random
w1 = np.random.rand(25, 28 * 28) - 0.5
b1 = np.random.rand(25, 1) - 0.5
w2 = np.random.rand(25, 25) - 0.5
b2 = np.random.rand(25, 1) - 0.5
w3 = np.random.rand(10, 25) - 0.5
b3 = np.random.rand(10, 1) - 0.5


def activation(x):
    return np.maximum(0, x)


def softmax(X):
    """
    Activation used for the final probalilities
    :param X:
    :return:
    """
    X = X - np.max(X, axis=0, keepdims=True)
    return np.exp(X) / np.sum(np.exp(X), axis=0, keepdims=True)


def forward(A0):
    """
    Forward propagation of the neural network
    :return: all hidden layers and the activated layers along with the output layer
    """
    A1 = w1.dot(A0) + b1
    Z1 = activation(A1)
    A2 = w2.dot(Z1) + b2
    Z2 = activation(A2)
    A3 = w3.dot(Z2) + b3
    Z3 = softmax(A3)
    return A1, Z1, A2, Z2, A3, Z3


def backwards(A0, A1, Z1, A2, Z2, A3, Z3):
    """
    Backwards propagation of the neural network
    :return: The partial derivatives of the loss function
    """
    m = train_y.size
    one_hot_y = one_hot(train_y, m)


def one_hot(Y, m):
    """
    Creates binary arrays where the i-th element of the array is a 1 when i = the label of Y
    :param Y: Labels for the MNIST dataset
    :param m: Number of elements in the dataset
    :return: the one hot encoded array
    """
    oh = np.zeros((m, np.max(Y) + 1))
    oh[np.arange(m), Y] = 1
    return oh


def grad_des():
    """
    The NN is controlled from here
    :return:
    """


if __name__ == '__main__':
    print(forward(train_X1.T)[5])
    # test_plot()
