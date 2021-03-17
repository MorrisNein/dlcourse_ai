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
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    
    pred_exp = np.exp(predictions - np.expand_dims(np.max(predictions, axis=-1), -1))
    probs = pred_exp/np.expand_dims(np.sum(pred_exp, axis=-1), -1)
    
    return probs

    # raise Exception("Not implemented!")


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
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    t_i = np.array(target_index, ndmin=1)
    t_i = t_i.reshape((t_i.shape[0], 1))
    
    prbs = np.array(probs, ndmin=2)

    loss = -np.sum(np.log(np.take_along_axis(prbs, t_i, axis=-1))) / prbs.shape[0]
    
    return loss

def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    t_i = np.array(target_index, ndmin=1)
    pred = np.array(predictions, ndmin=2)

    probs = softmax(pred)
    loss = cross_entropy_loss(probs, t_i)

    ground_truth = np.zeros_like(pred, dtype=int)
    ground_truth[np.arange(t_i.size), t_i.flatten()] = 1

    batch_size = len(t_i)
    
    dprediction = (probs - ground_truth) / batch_size
    dprediction = dprediction.reshape(predictions.shape)

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W
    # raise  Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)

    dW = np.einsum('ij,ik->jk', X, dprediction)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # raise Exception("Not implemented!")
            for batch_i in batches_indices:
              # loss = 0
              # dW = 0

              X_batch = X[batch_i]
              y_batch = np.vstack(y[batch_i])
              
              softmax_loss, dW_smx = linear_softmax(X_batch, self.W, y_batch)
              regularization_loss, dW_reg = l2_regularization(self.W, reg)

              dW = dW_smx + dW_reg
              self.W -= learning_rate * dW

            loss = softmax_loss + regularization_loss
            loss_history.append(loss)         

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))
            # print(f"Epoch {epoch}, loss_sm: {softmax_loss}, loss_reg: {regularization_loss},  W.shape: {self.W.shape}, dW.shape: {dW.shape}")

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        clf_output = X @ self.W
        probs = softmax(clf_output)
        y_pred = np.argmax(probs, axis=-1)

        return y_pred



                
                                                          

            

                
