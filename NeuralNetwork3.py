import sys
import numpy as np
import pandas as pd
import math


class Perceptron:
    # initialization
    def __init__(self, train_size, batch_size, num_H, learning_rate, epochs):
        self.train_size = train_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # assigning random weights to the layers of the perceptron
        self.W1 = np.random.randn(num_H, 784) * np.sqrt(1.0 / 784)
        self.B1 = np.random.randn(num_H, 1) * np.sqrt(1.0 / 784)
        self.W2 = np.random.randn(10, num_H) * np.sqrt(1.0 / num_H)
        self.B2 = np.random.randn(10, 1) * np.sqrt(1.0 / num_H)

    # Activation function
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        a = np.exp(-x)
        return 1.0 / (1.0 + a)

    def ReLU(self, x):
        return x * (x > 0)

    def dReLU(self, x):
        return 1. * (x > 0)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # forward propogation
    def forward_propogation(self, X):
        # layer 1
        Z1 = np.dot(self.W1, X) + self.B1
        A1 = self.ReLU(Z1)

        # layer 2
        Z2 = np.dot(self.W2, A1) + self.B2
        A2 = self.softmax(Z2)

        return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    # backward propogation
    def backward_propogation(self, X, Y, fp):
        m_d = 1 / self.train_size

        dZ2 = fp["A2"] - Y
        dW2 = np.dot(dZ2, fp["A1"].T) * m_d
        dB2 = np.sum(dZ2, axis=1, keepdims=True) * m_d

        dA1 = np.dot(self.W2.T, dZ2)

        dZ1 = dA1 * self.dReLU(fp["Z1"])
        dW1 = np.dot(dZ1, X.T) * m_d
        dB1 = np.sum(dZ1, axis=1, keepdims=True) * m_d

        return {'dw2': dW2, 'db2': dB2, 'dw1': dW1, 'db1': dB1}

    # utility to produce one hot matrix
    def one_hot(self, Y):
        one_hot_y = np.zeros((Y.size, 10))
        one_hot_y[np.arange(Y.size), Y] = 1

        return one_hot_y.T.astype(np.float64)

    # utility to get batches
    def get_batches(self, X, Y):
        num_of_batches = math.floor(self.train_size/self.batch_size)
        batches = []
        end_index = 0

        for i in range(num_of_batches):
            batches.append((X[:, i * self.batch_size: (i+1) * self.batch_size],
                           Y[:, i * self.batch_size: (i+1) * self.batch_size]))
            end_index = (i+1) * self.batch_size

        if self.train_size % self.batch_size != 0:
            batches.append((X[:, end_index:], Y[:, end_index:]))
        return batches

    # utility to shuffle a matrix
    def shuffle_array(self, X, Y):
        permutation = np.random.permutation(X.shape[1])
        X_train_shuffled = X[:, permutation]
        Y_train_shuffled = Y[:, permutation]

        return X_train_shuffled, Y_train_shuffled

    # utility to update parameters
    def update_parameters(self, bp_result):
        # upadate weights
        self.W1 = self.W1 - self.learning_rate * bp_result['dw1']
        self.W2 = self.W2 - self.learning_rate * bp_result['dw2']

        # update biases
        self.B1 = self.B1 - self.learning_rate * bp_result['db1']
        self.B2 = self.B2 - self.learning_rate * bp_result['db2']

    # training utility
    def train(self, X, Y):
        for i in range(self.epochs):
            shuffled_x, shuffled_y = self.shuffle_array(X, Y)

            batches = self.get_batches(shuffled_x, shuffled_y)

            for batch in batches:
                fp_results = self.forward_propogation(batch[0])
                bp_results = self.backward_propogation(
                    batch[0], batch[1], fp_results)
                self.update_parameters(bp_results)


# Driver program
if __name__ == "__main__":
    output_file = "test_predictions.csv"

    # getting input files - transposing and converting it to np arrays
    train_x = pd.read_csv(sys.argv[1], header=None).T.values / 255
    train_y = pd.read_csv(sys.argv[2], header=None).T.values
    test_x = pd.read_csv(sys.argv[3], header=None).T.values / 255

    network = Perceptron(train_x.shape[1], 32, 256, 0.3, 250)

    # vectorizations
    one_hot_y = network.one_hot(train_y)

    # train the model
    network.train(train_x, one_hot_y)

    # get predictions from the trained model
    predictions = np.argmax(network.forward_propogation(test_x)['A2'], 0)

    # write the predictions in the output file
    pd.DataFrame(predictions).to_csv(
        output_file, header=None, index=None)
