import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# index = 25
# plt.imshow(train_set_x_orig[index])

#print("y=" + str(train_set_y[:,index]) + ", it is a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + " picture")

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

# print("number_train: m_train = " + str(m_train))
# print("number_test: m_test = " + str(m_test))
# print("weight/high of picture: num_px = " + str(num_px))
# print("size of picture: " + str(num_px) + "," + str(num_px) + ", 3)")
# print("dim of train picture: " + str(train_set_x_orig.shape))
# print("dim of train tag: " + str(train_set_y.shape))
# print("dim of test picture: " + str(test_set_x_orig.shape))
# print("dim of test tag: " + str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# print("训练集降维后的维度： " + str(train_set_x_flatten.shape))
# print("训练集标签的维度： "+ str(train_set_y.shape))
# print("测试集降维后的维度： " + str(test_set_x_flatten.shape))
# print("测试集标签的维度： " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmod(z):
    s = 1/(1+np.exp(-z))
    return s
#train_set_x_orig[index]

#print(str(sigmod(9.2)))


def initialize_with_zeros(dim):
    w = np.zeros(shape = (dim,1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return (w, b)

def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmod(np.dot(w.T, X) + b)
    cost = (-1/m) * np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))

    dw = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {
        "dw": dw,
        "db": db
    }
    return (grads, cost)

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs=[]
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w-learning_rate * dw
        b = b-learning_rate * db

        if i%100 == 0:
            costs.append(cost)

        if (print_cost) and (i%100 == 0):
            print("迭代次数:%i, 误差值:%f" %(i, cost))

    params = {
        "w" : w,
        "b" : b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return (params, grads, costs)

# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1,0]])
# params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
# print(str(params["w"]))
# print(str(params["b"]))
# print(str(grads["dw"]))
# print(str(grads["db"]))

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmod(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction

# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1,0]])
# print("preditions = " + str(predict(w, b, X)))

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w, b = parameters["w"], parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("训练集精准度:" , format(100-np.mean(np.abs(Y_prediction_train- Y_train))*100), "%")
    print("测试集精准度：", format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100), "%")

    d={
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate = "+ str(d["learning_rate"]))
plt.show()
