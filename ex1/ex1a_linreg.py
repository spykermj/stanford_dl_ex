#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import time


def cost_gradient(theta, x_values, y_values):
    M = x_values.shape[1]
    y_hat = np.dot(theta.transpose(), x_values)
    cost = np.sum(np.square(y_hat - y_values)) / (2.0 * M)
    gradient = np.dot(x_values, (y_hat - y_values).transpose()) / M
    return (cost, gradient)


def main():
    training_filename = 'housing_training.data'
    test_filename = 'housing_test.data'
    train_data = np.loadtxt(training_filename)
    test_data = np.loadtxt(test_filename)

    train_data = train_data.transpose()
    test_data = test_data.transpose()

    # create a one valued intercept feature
    train_data = np.vstack((np.ones((1,train_data.shape[1])), train_data))
    test_data = np.vstack((np.ones((1,test_data.shape[1])), test_data))


    train_x = train_data[0:-1,:]
    train_y = train_data[-1,:]

    test_x = test_data[0:-1,:]
    test_y = test_data[-1,:]

    m = train_x.shape[1]
    n = train_x.shape[0]

    theta = np.random.random((n,1))

    start = time.clock()
    theta = minimize(cost_gradient, theta, args=(train_x, train_y), jac=True, options={'gtol': 1e-6, 'disp': True})['x']
    elapsed = time.clock()
    print("elapsed seconds for training {}".format(elapsed - start))

    actual_prices = train_y
    predicted_prices = np.dot(theta, train_x)

    train_rms = np.sqrt(np.mean((predicted_prices - actual_prices)**2))
    print('RMS training error: {}'.format(train_rms));

    actual_prices = test_y
    predicted_prices = np.dot(theta, test_x)

    test_rms = np.sqrt(np.mean((predicted_prices - actual_prices)**2))
    print('RMS testing error: {}'.format(test_rms));

    order_indices = np.argsort(actual_prices)

    actual_prices = [actual_prices[i] for i in order_indices]
    predicted_prices = [predicted_prices[i] for i in order_indices]

    plt.scatter(np.arange(len(actual_prices)), actual_prices, color='red', label='actual')
    plt.scatter(np.arange(len(predicted_prices)), predicted_prices, color='blue', label='prediction')
    plt.title('House Price Prediction')
    plt.xlabel('House number')
    plt.ylabel('House price ($1000s)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
