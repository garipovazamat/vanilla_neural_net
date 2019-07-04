import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu1 = ReLULayer()
        self.relu2 = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.relu1.zero_grad()
        
        X_epoch = self.layer1.forward(X)
        X_epoch = self.relu1.forward(X_epoch)
        X_epoch = self.layer2.forward(X_epoch)
        loss, grad = softmax_with_cross_entropy(X_epoch, y)
        
        grad = self.layer2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.layer1.backward(grad)

        return loss, grad

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        
        X_epoch = self.layer1.forward(X)
        X_epoch = self.relu1.forward(X_epoch)
        X_epoch = self.layer2.forward(X_epoch)
        return np.argmax(X_epoch, 1)
        

    def params(self):
        result = {'X_layer1': self.layer1.X, 'X_layer2': self.layer2.X,
                  'W_layer1': self.layer1.W, 'W_layer2': self.layer2.W,
                  'B_layer1': self.layer1.B, 'B_layer2': self.layer2.B,
                 }

        return result
