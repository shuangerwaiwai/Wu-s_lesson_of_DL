import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


np.random.seed(1)

X,Y = load_planar_dataset()

#plt.scatter(X[0,:], X[1,:], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]

# print("X的维度：" + str(shape_X))
# print("Y的维度：" + str(shape_Y))
# print("数据集里的数据有" + str(m) + "个")

# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)
#
# plot_decision_boundary(lambda  x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# LR_predictions = clf.predict(X.T)
# print("逻辑回归的准确性： %d " %float((np.dot(Y, LR_predictions)+np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100)+"% "
#       +"(正确标记的数据点所占的百分比)")


def layer_size(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros(shape=(n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {
        "W1": W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return (A2, cache)

# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1-A2))
    cost = -np.sum(logprobs) /m
    cost = float(np.squeeze(cost))

    assert(isinstance(cost, float))

    return cost

# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2-Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads

# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print("dW1=" + str(grads["dW1"]))
# print("dW2=" + str(grads["dW2"]))
# print("db1=" + str(grads["db1"]))
# print("db2=" + str(grads["db2"]))

def update_parameters(parameters, grads, learning_rate=1.2):
    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]

    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }

    return parameters

def nn_model(X, Y, n_h, num_iterations, print_cost = False):
    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=0.5)

        if print_cost:
            if i%1000 == 0:
                print("第 " , i, "次循环， 成本为： " + str(cost))

    return parameters

# X_assess, Y_assess = nn_model_test_case()
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost = False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions

parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)

plot_decision_boundary(lambda  x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print("准确率: %d" %float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100)+"%")

