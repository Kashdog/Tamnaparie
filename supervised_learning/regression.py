from __future__ import print_function, division
import numpy as np
import math

class LinearRegression(object):
    """ Base regression model. Models the relationship between a scalar dependent variable y and the independent 
    variables X. 
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations=100, learning_rate=0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """ Initialize weights randomly [-1/N, 1/N]
        w -- weights """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, )) 

    def fit(self, X, y):
        """ Insert constant ones for bias weights """
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        for i in range(self.n_iterations):
            """ See h(x): LinearRegression.png
            self.w = Theta -^ i
            X = x -^ i
            Dot product sums the the product of each weight(theta) times the input(x)

            h,theta(x^i)
            """

            y_pred = X.dot(self.w) 

            """Calculate Mean Squared Error or L2 Loss Function
            # See MSE.png for Equation
            y = y^i
            y_pred = h,theta(x^i) or mx^i
            np.mean = 1/N * Sigma"""

            mse = np.mean(0.5 * (y - y_pred)**2)
            self.training_errors.append(mse)

            """ Least Squares Cost Function:
            See LMS_one_example.png
            y_pred = h,theta(x)
            y = y
            .dot(X) = x -^j
            """

            grad_w = (y_pred - y).dot(X)

            """ Update the Weights: Gradient Descent.png
            learning_rate = alpha
            grad_w = derivative of J -^ Theta
             """
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        """ Insert constant ones for bias weights """
        X = np.insert(X, 0, 1, axis=1)
        """ 
        Calculate htheta(x) with updated weight
        Dot product, sums the the product of each weight(theta) times the input(x)
        h,theta(x^i)
        """

        y_pred = X.dot(self.w)
        return y_pred