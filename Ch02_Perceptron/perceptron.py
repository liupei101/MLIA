# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def parameter_initial(nx):
	perceptron_num = 1
	# initialize for w(nx, 1)
	w = np.zeros((nx, perceptron_num))
	# initialize for b
	b = np.zeros((perceptron_num, 1))
	return w, b

def train(X, y, learning_rate = 1, iterator_num = 100, visiable = False):
	# X.shape
	nx = X.shape[0]
	m = X.shape[1]
	assert y.shape == (1, m)
	# initialize
	w, b = parameter_initial(nx)
	# choose one point separating incorrectly
	finished = False
	cnt = 0
	while (finished is False) and (cnt <= iterator_num):
		finished = True
		for i in range(m):
			xi = X[:, i].reshape((nx, 1))
			yi = y[:, i].reshape((1, 1))
			if yi * (np.dot(w.T, xi) + b) <= 0:
				w += learning_rate * yi * xi
				b += learning_rate * yi
				finished = False
				# print the process
				if visiable:
					print "In %dth iterating: %dth point is seperated incorrectly" % (cnt+1, i+1)
					print "W:", w
					print "B:", b
				break
		cnt += 1

	return {"w" : w, "b" : b}

def predict(w, b, X):
	m = X.shape[1]
	# calculate y_hat
	y_hat = np.dot(w.T, X) + b
	y_hat[y_hat >= 0] = 1
	y_hat[y_hat < 0] = -1
	return y_hat

def load_data_xor():
	X_train = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]])
	y_train = np.array([[-1., 1., -1., 1.]])
	X_test = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]])
	y_test = np.array([[-1., 1., -1., 1.]])
	return X_train, y_train, X_test, y_test

def load_data():
	X_train = np.array([[3., 4., 1.], [3., 3., 1.]])
	y_train = np.array([[1., 1., -1.]])
	X_test = np.array([[3., 4., 1.], [3., 3., 1.]])
	y_test = np.array([[1., 1., -1.]])
	return X_train, y_train, X_test, y_test

if __name__ == '__main__':
	X_train, y_train, X_test, y_test = load_data_xor()
	# training data
	parameter = train(X_train, y_train, learning_rate = 1, visiable = True)
	# predicting data
	result = predict(parameter["w"], parameter["b"], X_test)
	print "origin  :", y_test
	print "predict :", result