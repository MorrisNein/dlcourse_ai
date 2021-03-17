import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax

from collections import OrderedDict

class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = OrderedDict({
          "linear_1": FullyConnectedLayer(n_input, hidden_layer_size),
          "relu_1": ReLULayer(),
          "linear_2": FullyConnectedLayer(hidden_layer_size, n_output)
          })

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params = self.params().values()
        for par in params:      
          par.grad = np.zeros_like(par.grad)  

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        # Forward pass
        for n_lay, lay in enumerate(self.layers.values()):
          if n_lay == 0:
            # print(X)
            current_X = lay.forward(X)
          else:
            # print(current_X)
            current_X = lay.forward(current_X)
          # print(current_X)
          # print(f"{n_lay}, {lay}")

        clf_output = current_X
        
        CE_loss, dpredictions = softmax_with_cross_entropy(clf_output, y)
        
        # Backward pass
        for n_lay, lay in enumerate(reversed(self.layers.values())):
          # print(f"{n_lay}")
          if n_lay == 0:
            # print(dpredictions)
            current_dX = lay.backward(dpredictions)
          else:
            # print(current_dX)
            current_dX = lay.backward(current_dX)
          # print(current_dX)
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        reg_loss_accumulated = 0
        for par in params:      
          reg_loss, dpar = l2_regularization(par.value, self.reg)
          par.grad += dpar
          reg_loss_accumulated += reg_loss

        loss = CE_loss + reg_loss_accumulated
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        y_pred = np.zeros(X.shape[0], np.int)
        for n_lay, lay in enumerate(self.layers.values()):
          if n_lay == 0:
            current_X = lay.forward(X)
          else:
            current_X = lay.forward(current_X)
        clf_output = current_X
        
        probs = softmax(clf_output)
        # print(probs)
        y_pred = np.argmax(probs, axis=-1)
        # print(y_pred)
        # raise Exception("Not implemented!")
        return y_pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for layname, lay in self.layers.items():
          result.update({f"{layname}_{parname}" : par for parname, par in lay.params().items()})

        return result

class OneLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = OrderedDict({
          "linear_1": FullyConnectedLayer(n_input, n_output),
          })

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params = self.params().values()
        for par in params:      
          par.grad = np.zeros_like(par.grad)  

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        # Forward pass
        for n_lay, lay in enumerate(self.layers.values()):
          if n_lay == 0:
            current_X = lay.forward(X)
          else:
            current_X = lay.forward(current_X)
        clf_output = current_X
        
        CE_loss, dpredictions = softmax_with_cross_entropy(clf_output, y)
        
        # Backward pass
        for n_lay, lay in enumerate(reversed(self.layers.values())):
          if n_lay == 0:
            current_dX = lay.backward(dpredictions)
          else:
            current_dX = lay.backward(current_dX)
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        reg_loss_accumulated = 0
        for par in params:      
          reg_loss, dpar = l2_regularization(par.value, self.reg)
          par.grad += dpar
          reg_loss_accumulated += reg_loss

        loss = CE_loss + reg_loss_accumulated
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        y_pred = np.zeros(X.shape[0], np.int)
        for n_lay, lay in enumerate(self.layers.values()):
          if n_lay == 0:
            current_X = lay.forward(X)
          else:
            current_X = lay.forward(current_X)
        clf_output = current_X
        
        probs = softmax(clf_output)
        # print(probs)
        y_pred = np.argmax(probs, axis=-1)
        # print(y_pred)
        # raise Exception("Not implemented!")
        return y_pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for layname, lay in self.layers.items():
          result.update({f"{layname}_{parname}" : par for parname, par in lay.params().items()})

        return result
