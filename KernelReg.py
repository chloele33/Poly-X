# -*- coding: utf-8 -*-

import numpy as np
from math import exp,sqrt,pi
import pandas as pd

import sklearn
from sklearn.linear_model import SGDClassifier

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin


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










