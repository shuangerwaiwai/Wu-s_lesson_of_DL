import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils
import reg_utils
import gc_utils

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=True)

def model(X, Y, learning_rate = 0.01, num_iteration=15000, print_cost = True, initialization="he", is_plot = True):
    grads={}
    costs=[]
    m = X.shape[1]
    layers_dim = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dim)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dim)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dim)
    else:
        print("error initialize!exit!")
        exit

    for i in range(0, num_iteration):
        a3, cache = init_utils.forward_propagation(X, parameters)

        cost = init_utils.compute_loss(a3, Y)

        grads = init_utils.backward_propagation(X, Y, cache)

        parameters = init_utils.update_parameters(parameters, grads, learning_rate)

        if i%1000 == 0:
            costs.append(cost)
            if print_cost:
                print(str(i) + "th iteration, cost is:" + str(cost))

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('ietrations (per hundreds)')
        plt.title('learning rate = ' + str(learning_rate))
        plt.show()

    return parameters
def initialize_parameters_zeros(layers_dims):
    parameters={}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

# parameters = initialize_parameters_zeros([3,2,1])
# print("W1=" + str(parameters["W1"]))
# print("b1=" + str(parameters["b1"]))
# print("W2=" + str(parameters["W2"]))
# print("b2=" + str(parameters["b2"]))

#parameters = model(train_X, train_Y, initialization= "zeros", is_plot=True)

# print("train_set:")
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
#
# print("test_set:")
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
#
# print("prediction_train = " + str(predictions_train))
# print("prediction_test = " + str(predictions_test))
#
# plt.title("Model with Zeros initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*10
        parameters["b" + str(l)] = np.random.randn(layers_dims[l], 1)

        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

# parameters = initialize_parameters_random([3, 2, 1])
# print("W1=" + str(parameters["W1"]))
# print("b1=" + str(parameters["b1"]))
# print("W2=" + str(parameters["W2"]))
# print("b2=" + str(parameters["b2"]))

# parameters = model(train_X, train_Y, initialization="random", is_plot=True)
# print("train_set:")
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
#
# print(predictions_train)
# print(predictions_test)
#
# plt.title("Model with large random initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters={}
    L = len(layers_dims)

    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])* np.sqrt(2/layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

# parameters = initialize_parameters_random([2, 4, 1])
# print("W1=" + str(parameters["W1"]))
# print("b1=" + str(parameters["b1"]))
# print("W2=" + str(parameters["W2"]))
# print("b2=" + str(parameters["b2"]))

# parameters = model(train_X, train_Y, initialization="he", is_plot=True)
# print("train_set:")
# prediction_train = init_utils.predict(train_X, train_Y, parameters)
# print("test_set:")
# prediction_test = init_utils.predict(test_X, test_Y, parameters)
#
# plt.title("Model with He initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)


train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=True)

def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, is_plot = True, lambd=0, keep_prob=1):
    grads={}
    costs=[]
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    parameters = reg_utils.initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        if keep_prob == 1:
            a3, cache = reg_utils.forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        else:
            print("keep_prob error! exit!")
            exit

        if lambd == 0:
            cost = reg_utils.compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        assert(lambd == 0 or keep_prob == 1)

        if (lambd == 0 and keep_prob == 1):
            grads = reg_utils.backward_propagation(X, Y, cache)
        elif lambd!=0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob<1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        parameters = reg_utils.update_parameters(parameters, grads, learning_rate)

        if i % 1000 == 0:
            costs.append(cost)
            if(print_cost and i % 10000 == 0):
                print(str(i) + "th iteration, cost is:" + str(cost))

    if is_plot:
        plt.plot(costs)
        plt.ylabel("cost")
        plt.xlabel("iteration (x1,000)")
        plt.title("Learning rate = "+ str(learning_rate))
        plt.show()

    return parameters

# parameters = model(train_X, train_Y, is_plot=True)
# print("train set:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("test set:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#
# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = reg_utils.compute_cost(A3, Y)
    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))/(2*m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def backward_propagation_with_regularization(X, Y, cache, lambd):

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = (1/m) * np.dot(dZ3, A2.T) + ((lambd * W3)/m)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2>0))
    dW2 = (1/m) * np.dot(dZ2, A1.T) + ((lambd * W2)/m)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = (1/m) * np.dot(dZ1, X.T) + ((lambd * W1)/m)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

# parameters = model(train_X, train_Y, lambd=0.7, is_plot=True)
# print("train_set:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("test set:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#
# plt.title("Model with L2-regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = reg_utils.relu(Z1)

    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 = A1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = reg_utils.relu(Z2)

    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3 = A3 - Y
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dZ2 = np.multiply(dA2, np.int64(A2>0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)

    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

# parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3, is_plot=True)
# print("dropout, train_set:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("dropout, test_set:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#
# plt.title("Model with dropout")
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

def forward_propagation_n(X, Y, parameters):
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = gc_utils.relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = gc_utils.relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = gc_utils.sigmoid(Z3)

    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1-A3), 1-Y)
    cost = (1/m) * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache

def backward_propagation_n(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1./m) * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2>0))
    dW2 = 1./m*np.dot(dZ2, A1.T)
    db2 = 1./m*np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = 1./m*np.dot(dZ1, X.T)
    db1 = 1./m*np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    parameters_value, keys = gc_utils.dictionary_to_vector(parameters)
    grad = gc_utils.gradients_to_vector(gradients)
    num_parameters = parameters_value.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        thetaplus = np.copy(parameters_value)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        J_plus[i], cache = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(thetaplus))

        thetaminus = np.copy(parameters_value)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        J_minus[i], cache = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(thetaminus))

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference < 1e-7:
        print("yes!")
    else:
        print("No!")

    return difference
