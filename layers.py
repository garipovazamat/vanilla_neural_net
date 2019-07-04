import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        max_pred = np.max(predictions)
        preds = predictions - max_pred
    else:
        max_pred = np.max(predictions, 1)
        max_pred = max_pred.reshape((max_pred.shape[0], 1)).repeat(predictions.shape[1], 1)
    
    preds = predictions - max_pred
    
    if len(preds.shape) > 1:
        return np.array([np.exp(predsi) / np.sum(np.exp(predsi)) for predsi in preds])
        
    return np.exp(preds) / np.sum(np.exp(preds))


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    px = get_target_neurons(target_index, probs.shape)
    
    return -1 * np.sum(px * np.log(probs))


def get_target_neurons(target_index, size):
    px = np.zeros(size)
    if (type(target_index) == int):
        px[target_index] = 1
    else:
        for row, i in enumerate(target_index):
            px[row][i] = 1
            
    return px


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    softmax_out = softmax(preds)
    loss = cross_entropy_loss(softmax_out, target_index)
    target_neurons = get_target_neurons(target_index, softmax_out.shape)
    dPredictions = softmax_out - target_neurons
    return loss, dPredictions


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.last_grad = None

    def forward(self, X):
        relu = np.array([[np.max((0, e)) for e in x] for x in X])
        self.last_grad = np.array([[1 if e > 0 else 0 for e in relu_x] for relu_x in relu])
        return relu

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = self.last_grad * d_out
        return d_result

    def params(self):
        return {}
    
    def zero_grad(self):
        if self.last_grad is not None:
            self.grad = np.zeros_like(self.last_grad)


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = Param(X)
        xw = X.dot(self.W.value) + self.B.value
        return xw

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        self.W.grad = self.X.value.transpose().dot(d_out)
        self.X.grad = d_out.dot(self.W.value.transpose())
        self.B.grad = np.array([np.sum(d_out, axis=0).transpose()])

        return self.X.grad

    def params(self):
        return {'W': self.W, 'B': self.B}
    
    def zero_grad(self):
        if self.X is not None:
            self.X.grad = np.zeros_like(self.X.value)
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
