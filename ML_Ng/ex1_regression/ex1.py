import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def getData(filename):
	f = open(filename,"r")
	x, y = [], []
	for line in f.readlines():
		x.append(eval(line.split(',')[0]))
		y.append(eval(line.split(',')[1].strip()))
	f.close()
	m = len(x)
	x = np.array(x)
	y = np.array(y)
	return x, y, m

def gradientDescent(x, y, m, theta, alpha, iterations):
	h = theta[0] + theta[1] * x
	for i in range(iterations):
	    p = theta[0] - (alpha/m) * sum(h - y)
	    q = theta[1] - (alpha/m) * sum((h - y) * x)
	    theta[0] = p
	    theta[1] = q
	    h = theta[0] + theta[1] * x
	J = sum((h - y)**2)/(2*m)
	print("Cost: {:.3f}".format(J))
	return theta, h

def plotData(x, y, h):
	title = "Figure 1: Plot of training data and linear regression"
	plt.figure(title)
	plt.plot(x, y, 'rx')
	plt.plot(x, h, 'b-')
	plt.xlabel('Profit in $10,000s')
	plt.ylabel('Population of City in 10,000s')
	plt.legend(labels=["Training data", "Linear regression"])
	plt.title(title)
	plt.show()

def plot3Dfig(x, y, m, theta):
	n = 100
	theta0_vals = np.linspace(-10, 10, n)
	theta1_vals = np.linspace(-1, 4, n)

	J_vals = np.ones((n, n))
	for i in range(n):
		for j in range(n):
			h = theta0_vals[i] + theta1_vals[j] * x
			J_vals[i][j] = (sum((h-y)**2/(2*m)))
	J_vals = J_vals.transpose()

	'''
	Jlist = []
	for i in range(n):
	    for j in range(n):
	        h = theta0_vals[j] + theta1_vals[i] * x
	        Jlist.append(sum((h-y)**2/(2*m)))
	J_vals = np.array(Jlist).reshape(n, n)
	J_vals = np.transpose(J_vals)
	'''

	title2 = "Figure 2: Surface of cost function"
	fig = plt.figure(title2)
	ax = Axes3D(fig)
	X, Y = np.meshgrid(theta0_vals, theta1_vals)
	ax.plot_surface(X, Y, J_vals)
	plt.xlabel('theta0')
	plt.ylabel('theta1')
	plt.title(title2)
	plt.show()

	title3 = "Figure 3: Contour of cost function"
	plt.figure(title3)
	plt.contour(X, Y, np.log(J_vals))
	plt.plot(theta[0], theta[1], 'rx')
	plt.xlabel('theta0')
	plt.ylabel('theta1')
	plt.title(title3)
	plt.show()

def main():
	filename = "ex1data1.txt"
	theta = [0, 0]
	alpha = 0.02
	iterations = 1000
	x, y, m = getData(filename)
	theta, h = gradientDescent(x, y, m, theta, alpha, iterations)
	print(theta)
	print("linear regression : h = {:.3f} + {:.3f}*x".format(theta[0], theta[1]))
	plotData(x, y, h)
	plot3Dfig(x, y, m, theta)

main()
