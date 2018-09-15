# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Creates a Dataset to be used for Linear Regression
from sklearn.datasets import make_regression
#Import Data Processing and Visualization utilities
from utils import train_test_split, mean_squared_error
from supervised_learning import LinearRegression
def main():
    """ Create a Dataset """
    X, y = make_regression(n_samples=100, n_features=1, noise=20)

    """ Split into Train and Test with test containing 2/5 of data """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    """ Obtain number of samples and features from the shape of X"""
    n_samples, n_features = np.shape(X)
     
    """ Initialize Linear Regression Model with 100 training iterations """
    model = LinearRegression(n_iterations=100)

    """ Fit the model to the data """
    model.fit(X_train, y_train)
    
    """ Plot the Model Training Errors """
    n = len(model.training_errors)
    training, = plt.plot(range(n), model.training_errors, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()

    """ Predict for the Test Data and Calculate the MSE """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean squared error: %s" % (mse))

    """ Predict the Line for all the data """
    y_pred_line = model.predict(X)

    """ Color Map """
    cmap = plt.get_cmap('viridis')

    """ Plot the results"""
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Linear Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()