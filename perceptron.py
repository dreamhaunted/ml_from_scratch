import numpy as np

class Perceptron(object):
    """
    Parameters:
    ------------
    Eta - learning rate
    Epochs - number of iterations
    Random_state - specialize it in order to have same weights every time

    Attributes:
    -----------
    w_ - weight vector
    errors_ - vector of errors

    Methods:
    -----------
    fit - initialize weights and error list, split the data, calculate errors and update the attrs
    z - calculate the dot product
    predict - threshold behavior: returns -1/1 for bad/good prediction
    """
    def __init__(self, eta=1, epochs=50, random_state=42):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Parameters:
        -----------
        X - a matrix of values  | shape = (n_values, n_features)
        y - a vector of targets | shape = (n_values)
        """

        self.errors_ = []
        self.w_ = abs(np.random.randn(X.shape[1] + 1, random_state=self.random_state))

        for _ in range(self.epochs):
            errors = 0
            for value, target in zip(X, y):
                delta = self.eta * (target - self.predict(value))
                self.w_[0] += update
                self.w_[1:] += update * value
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def z(self, X):
        return np.dot(X, w_[1:]) + w[0]

    def predict(self):
        return np.where(self.z(X) >= 0.0, 1, -1)
