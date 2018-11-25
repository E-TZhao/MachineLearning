import numpy as np
import scipy.io as sio
import scipy.optimize as op
import matplotlib.pyplot as plt

def getData(filename):
	data = sio.loadmat(filename)
	#weights = sio.loadmat("ex4dweights.mat")
	x = data['X']
	y = data['y'].reshape(x.shape[0], )
	return x, y

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def sigmoidGradient(z):
	return sigmoid(z) * (1-sigmoid(z))

def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, L_in + 1) * 2*epsilon_init - epsilon_init
    return W

def unrollingParams(Theta1, Theta2):
	return np.append(Theta1.flatten(), Theta2.flatten())

def rollingParams(nn_params, input_layer_size, hidden_layer_size, num_labels):

	Theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size, input_layer_size+1)
	Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels, hidden_layer_size+1)
	return Theta1, Theta2

'''
def feedforward(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lam):
	m = x.shape[0]
	X = np.hstack((np.ones((m, 1)), x))
	Y = np.zeros((m, num_labels))
	for i in range(m):
		if y[i] == 10:
			Y[i, 0] = 1
		else:
			Y[i, y[i]] = 1
	Theta1, Theta2 = rollingParams(nn_params, input_layer_size, hidden_layer_size, num_labels)
	z2 = np.dot(X, Theta1.T)
	a2 = sigmoid(z2)           
	a2 = np.hstack((np.ones((a2.shape[0],1)), a2))
	z3 = np.dot(a2, Theta2.T)
	a3 = sigmoid(z3)
	h = a3
'''

# Three layers(one input layer, one hidden layer, one output layer)
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lam):
	m = x.shape[0]
	X = np.hstack((np.ones((m, 1)), x))
	Y = np.zeros((m, num_labels))
	for i in range(m):
		if y[i] == 10:
			Y[i, 0] = 1
		else:
			Y[i, y[i]] = 1
	Theta1, Theta2 = rollingParams(nn_params, input_layer_size, hidden_layer_size, num_labels)
	z2 = np.dot(X, Theta1.T)
	a2 = sigmoid(z2)           
	a2 = np.hstack((np.ones((a2.shape[0],1)), a2))
	z3 = np.dot(a2, Theta2.T)
	a3 = sigmoid(z3)
	h = a3

	J = (1/m) * np.sum(-Y * np.log(h) - (1-Y) * np.log(1-h)) \
		+ (lam/(2*m))*(np.sum(Theta1**2) + np.sum(Theta2**2))      # 有问题,theta[:,0]不应该正则化
	return J


def gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lam):
	m = x.shape[0]
	X = np.hstack((np.ones((m, 1)), x))
	Y = np.zeros((m, num_labels))
	for i in range(m):
		if y[i] == 10:
			Y[i, 0] = 1
		else:
			Y[i, y[i]] = 1
	Theta1, Theta2 = rollingParams(nn_params, input_layer_size, hidden_layer_size, num_labels)
	z2 = np.dot(X, Theta1.T)
	a2 = sigmoid(z2)           
	a2 = np.hstack((np.ones((a2.shape[0],1)), a2))
	z3 = np.dot(a2, Theta2.T)
	a3 = sigmoid(z3)
	h = a3

	delta3 = h - Y
	delta2 = np.dot(delta3, Theta2) * np.hstack((np.ones((m, 1)), sigmoidGradient(z2)))
	delta2 = delta2[:, 1:]
	theta2_grad = np.dot(delta3.T, a2) / m
	theta2_grad[:, 1:] = theta2_grad[:, 1:] + (lam/m)*Theta2[:, 1:]
	theta1_grad = np.dot(delta2.T, X) / m
	theta1_grad[:, 1:] = theta1_grad[:, 1:] + (lam/m)*Theta1[:, 1:]
	grad = unrollingParams(theta1_grad, theta2_grad)
	return grad


def predictAccuracy(Theta1, Theta2, x, y):
	m = x.shape[0]
	X = np.hstack((np.ones((m, 1)), x))
	z2 = np.dot(X, Theta1.T)
	a2 = sigmoid(z2)           
	a2 = np.hstack((np.ones((a2.shape[0],1)), a2))
	z3 = np.dot(a2, Theta2.T)
	a3 = sigmoid(z3)
	h = a3
	for i in range(m):
		if y[i] == 10:
			y[i] = 0
	p = h.argmax(axis = 1)
	accuracy = np.sum(p == y)/m
	return accuracy


#def main():
filename = "ex4data1.mat"
x, y = getData(filename)
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lam = 1
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
nn_params = unrollingParams(Theta1, Theta2)
#J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lam)
#print(J, grad.shape)

result = op.minimize(fun = nnCostFunction, x0 = nn_params, 
	args = (input_layer_size, hidden_layer_size, num_labels, x, y, lam), method = 'TNC', jac = gradient)

Theta1, Theta2 = rollingParams(result.x, input_layer_size, hidden_layer_size, num_labels)
accuracy = predictAccuracy(Theta1, Theta2, x, y)
print("Training Set Accuracy: {:%}".format(accuracy))

