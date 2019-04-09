# -*- coding: utf-8 -*-


import numpy as np
from math import exp,sqrt,pi
import pandas as pd

import sklearn
from sklearn.linear_model import SGDClassifier

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin

fourbyfour = np.array([
                       [1,2,3,4],
                       [3,2,1,4],
                       [5,4,6,7],
                       [11,12,13,14]
                      ])


twobyfourbythree = np.array([
                             [[2,3],[11,9],[32,21],[28,17]],
                             [[2,3],[1,9],[3,21],[28,7]],
                             [[2,3],[1,9],[3,21],[28,7]],
                            ])

print('4x4*4x2x3 matmul:\n {}\n'.format(np.matmul(-fourbyfour,twobyfourbythree)))
print('4x4*4x2x3 @:\n {}\n'.format(-fourbyfour @ twobyfourbythree ))



#Stat functions
def get_accuracy(Y_pred, Y_gold):
    accuracy = np.equal(Y_pred, Y_gold).sum() / len(Y_pred)
    return accuracy


def get_stats(Y_pred, Y_gold):
    TP = sum([1 if pred == 1 and gold == 1 else 0 for pred,gold in zip(Y_pred, Y_gold)])
    FP = sum([1 if pred == 1 and gold == 0 else 0 for pred,gold in zip(Y_pred, Y_gold)])
    FN = sum([1 if pred == 0 and gold == 1 else 0 for pred,gold in zip(Y_pred, Y_gold)])
    return TP, FP, FN


def get_precision(Y_pred, Y_gold):
    TP, FP, FN = get_stats(Y_pred, Y_gold)
    precision = TP / float(TP + FP)
    return precision


def get_recall(Y_pred, Y_gold):
    TP, FP, FN = get_stats(Y_pred, Y_gold)
    recall = TP / float(TP + FN)
    return recall


def get_f1(Y_pred, Y_gold):
    recall = get_recall(Y_pred, Y_gold)
    precision = get_precision(Y_pred, Y_gold)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# assuming using two points for bounding box
# assuming 2D points defined by (x, y)

a = [0, 3]
b = [2, 5]
c = [3, 0]
d = [7, 2]

MrrBoundingBox1 = np.array([])
np.append(MrrBoundingBox1, [0,3])
np.append(MrrBoundingBox1, [3, 0])

MrrBoundingBox2 = np.array([[2,5], [7, 2]])

SinkBoundingBox1 = np.array([[0, 6], [3, 4]])
SinkBoundingBox2 = np.array([[2, 5], [7, 3]])

SinkBoundingBoxTest1 = np.array([[2, 8], [6, 6]]);
WallTestLocation = [0, 10];
CeilingTestLocation = [12];

leftWall1 = -2
rightWall1 = 8


leftWall2 = -3
rightWall2 = 12

ceiling1 = -2
ceiling2 = 0




# extract feature sets: order: sink bounding lower x, sink bounding lower y,
# sink bounding top x, sink bounding top y,
# left wall distance, distance to top of sink , distance to sink left,
X_train_bottomLeft = [[0, 6, 3, 4, 2, 1, 0], [2, 5, 7, 3, 4, 1, 0]];
Y_train_bottomLeft= [a, b]
#a
#b
#c
#d


# ATTEMPT 2
# extract feature sets: order: sink bounding lower x, sink bounding lower y,
# sink bounding top x, sink bounding top y, leftwall, right wall

# left wall distance, distance to top of sink , distance to sink left,





X_train_bottomLeft2 = [[0, 6, 3, 4, -2, 8], [2, 5, 7, 3, -3, 12]]
Y_train_bottomLeft2 = [[2, 1, 0], [5, 1, 0]]
X_testSetBottomLeft2 = [[2, 8, 6, 6, 0, 10]]
Y_testSetBottomLeft2 = [[2, 1, 0]]




# Train Stochastic Gradient Descent SGD
def f_calcaulting_avg_cross_validation_accuracy(alpha):
    accuracy_sum = 0
    #for i in range(5):
    test_set_X = X_train_bottomLeft2
    test_set_y = Y_train_bottomLeft2
    # training_big_set_X = np.append(list_Xfold_arr[(i + 1) % 5], list_Xfold_arr[(i + 2) % 5], axis=0)
    # training_big_set_X = np.append(training_big_set_X, list_Xfold_arr[(i + 3) % 5], axis=0)
    # training_big_set_X = np.append(training_big_set_X, list_Xfold_arr[(i + 4) % 5], axis=0)
    # # append the other four fold sets' labels to get one big set of labels
    # training_big_set_y = np.append(list_yfold_arr[(i + 1) % 5], list_yfold_arr[(i + 2) % 5], axis=0)
    # training_big_set_y = np.append(training_big_set_y, list_yfold_arr[(i + 3) % 5], axis=0)
    # training_big_set_y = np.append(training_big_set_y, list_yfold_arr[(i + 4) % 5], axis=0)
    # run predict
    # run training functions
    clf = SGDClassifier(loss='log', penalty='l2', max_iter=1000, alpha = alpha)
    clf.fit(X_train_bottomLeft2, [1, 2])
    Y_pred = predict(clf, X_testSetBottomLeft2)
    print(Y_pred)
    curr_accuracy = get_accuracy(Y_pred, [1])
    accuracy_sum += curr_accuracy
    return accuracy_sum / 5

def tune_SDG():
    potential_alphas = [0.001, 0.004, 0.007, 0.008, 0.009, 0.01, 0.02]
    best_alpha = 0.0001
    best_cross_validation_accuracy = 0
    for alpha in potential_alphas:
        cross_valid_accuracy = f_calcaulting_avg_cross_validation_accuracy(alpha)
        if cross_valid_accuracy > best_cross_validation_accuracy:
            best_cross_validation_accuracy = cross_valid_accuracy
            best_alpha = alpha
    print("best alpha is:", best_alpha)


#after tuning, best alpha was 0.007
def SGD_fit(X_train, y_train):
    clf = SGDClassifier(loss='log', penalty='l2', max_iter=1000, alpha = 0.007)
    clf.fit(X_train, y_train)
    return clf

#predict function
def predict(clf, X_test):
    Y_pred = clf.predict(X_test)
    return Y_pred


tune_SDG();


#simple_SDG_Pa = simple_SDG_Pa()




# ATTEMPT KERNEL REGRESSION
class KernelRegression(BaseEstimator, RegressorMixin):
    def __init__(self, kernel="rbf", gamma=None):
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y):
        self.X = X
        self.y = y

        if hasattr(self.gamma, "__iter__"):
            self.gamma = self._tune_gamma(self.gamma)

        return self

    def predict(self, X):
        K = pairwise_kernels(self.X, X, metric=self.kernel, gamma=self.gamma)
        return (K * self.y[:, None]).sum(axis=0) / K.sum(axis=0)

    def _tune_gamma(self, gamma_values):
        # Select specific value of gamma from the range of given gamma_values
        # by minimizing mean-squared error in leave-one-out cross validation
        mse = np.empty_like(gamma_values, dtype=np.float)
        for i, gamma in enumerate(gamma_values):
            K = pairwise_kernels(self.X, self.X, metric=self.kernel,
                                 gamma=gamma)
            np.fill_diagonal(K, 0)  # leave-one-out
            Ky = K * self.y[:, np.newaxis]
            y_pred = Ky.sum(axis=0) / K.sum(axis=0)
            mse[i] = ((y_pred - self.y) ** 2).mean()

        return gamma_values[np.nanargmin(mse)]

    def fit2(test_X, train_X, train_y, bandwidth=1.0, kn='box'):
        kernels = {
            'box': lambda x: 1 / 2 if (x <= 1 and x >= -1) else 0,
            'gs': lambda x: 1 / sqrt(2 * pi) * exp(-x ** 2 / 2),
            'ep': lambda x: 3 / 4 * (1 - x ** 2) if (x <= 1 and x >= -1) else 0
        }
        predict_y = []
        for entry in test_X:
            nks = [np.sum((j - entry) ** 2) / bandwidth for j in train_X]
            ks = [kernels['box'](i) for i in nks]
            dividend = sum([ks[i] * train_y[i] for i in range(len(ks))])
            divisor = sum(ks)
            predict = dividend / divisor
            predict_y.extend(predict)
            # print(entry)
        return np.array(predict_y)[:, np.newaxis]


###################################Bishop's Kernel###################
import numpy as np
import matplotlib.pyplot as plt

from prml.kernel import (
    PolynomialKernel,
    RBF,
    GaussianProcessClassifier,
    GaussianProcessRegressor
)

def create_toy_data(func, n=10, std=1., domain=[0., 1.]):
    x = np.linspace(domain[0], domain[1], n)
    t = func(x) + np.random.normal(scale=std, size=n)
    return x, t

def sinusoidal(x):
        return np.sin(2 * np.pi * x)

#TRIAL 1 DUAL REPRESENTATION
x_train, y_train = create_toy_data(sinusoidal, n=10, std=0.1)
x = np.linspace(0, 1, 100)

model = GaussianProcessRegressor(kernel=PolynomialKernel(3, 1.), beta=int(1e10))
model.fit(x_train, y_train)


y = model.predict(x)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
plt.plot(x, y, color="r", label="gpr")
plt.show()


# GAUSSIAN PROCESSES
#TRIAL 2
x_train, y_train = create_toy_data(sinusoidal, n=7, std=0.1, domain=[0., 0.7])
x = np.linspace(0, 1, 100)

model = GaussianProcessRegressor(kernel=RBF(np.array([1., 15.])), beta=100)
model.fit(x_train, y_train)

y, y_std = model.predict(x, with_error=True)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
plt.plot(x, y, color="r", label="gpr")
plt.fill_between(x, y - y_std, y + y_std, alpha=0.5, color="pink", label="std")
plt.show()


#TRIAL 3
x_train, y_train = create_toy_data(sinusoidal, n=7, std=0.1, domain=[0., 0.7])
x = np.linspace(0, 1, 100)

print("xtrain", x_train)
print("ytrain", y_train)
print("x", x)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
model = GaussianProcessRegressor(kernel=RBF(np.array([1., 1.])), beta=100)
model.fit(x_train, y_train)
y, y_std = model.predict(x, with_error=True)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
plt.plot(x, y, color="r", label="gpr {}".format(model.kernel.params))
plt.fill_between(x, y - y_std, y + y_std, alpha=0.5, color="pink", label="std")
plt.legend()

plt.subplot(1, 2, 2)
model.fit(x_train, y_train, iter_max=100)
y, y_std = model.predict(x, with_error=True)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
plt.plot(x, y, color="r", label="gpr {}".format(np.round(model.kernel.params, 2)))
plt.fill_between(x, y - y_std, y + y_std, alpha=0.5, color="pink", label="std")
plt.legend()
plt.show()


#TRIAL 4
def create_toy_data_3d(func, n=10, std=1.):
    x0 = np.linspace(0, 1, n)
    x1 = x0 + np.random.normal(scale=std, size=n)
    x2 = np.random.normal(scale=std, size=n)
    t = func(x0) + np.random.normal(scale=std, size=n)
    return np.vstack((x0, x1, x2)).T, t

x_train, y_train = create_toy_data_3d(sinusoidal, n=20, std=0.1)
x0 = np.linspace(0, 1, 100)
x1 = x0 + np.random.normal(scale=0.1, size=100)
x2 = np.random.normal(scale=0.1, size=100)
x = np.vstack((x0, x1, x2)).T

model = GaussianProcessRegressor(kernel=RBF(np.array([1., 1., 1., 1.])), beta=100)
model.fit(x_train, y_train)
y, y_std = model.predict(x, with_error=True)
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_train[:, 0], y_train, facecolor="none", edgecolor="b", label="training")
plt.plot(x[:, 0], sinusoidal(x[:, 0]), color="g", label="$\sin(2\pi x)$")
plt.plot(x[:, 0], y, color="r", label="gpr {}".format(model.kernel.params))
plt.fill_between(x[:, 0], y - y_std, y + y_std, color="pink", alpha=0.5, label="gpr std.")
plt.legend()
plt.ylim(-1.5, 1.5)

model.fit(x_train, y_train, iter_max=100, learning_rate=0.001)
y, y_std = model.predict(x, with_error=True)
plt.subplot(1, 2, 2)
plt.scatter(x_train[:, 0], y_train, facecolor="none", edgecolor="b", label="training")
plt.plot(x[:, 0], sinusoidal(x[:, 0]), color="g", label="$\sin(2\pi x)$")
plt.plot(x[:, 0], y, color="r", label="gpr {}".format(np.round(model.kernel.params, 2)))
plt.fill_between(x[:, 0], y - y_std, y + y_std, color="pink", alpha=0.2, label="gpr std.")
plt.legend()
plt.ylim(-1.5, 1.5)
plt.show()


#TRIAL 5
def create_toy_data():
    x0 = np.random.normal(size=50).reshape(-1, 2)
    x1 = np.random.normal(size=50).reshape(-1, 2) + 2.
    return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)[:, None]

x_train, y_train = create_toy_data()
x0, x1 = np.meshgrid(np.linspace(-4, 6, 100), np.linspace(-4, 6, 100))
x = np.array([x0, x1]).reshape(2, -1).T

model = GaussianProcessClassifier(RBF(np.array([1., 7., 7.])))
model.fit(x_train, y_train)
y = model.predict(x)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.ravel())
plt.contourf(x0, x1, y.reshape(100, 100), levels=np.linspace(0,1,3), alpha=0.2)
plt.colorbar()
plt.xlim(-4, 6)
plt.ylim(-4, 6)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()






