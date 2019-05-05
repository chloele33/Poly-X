# -*- coding: utf-8 -*-


import numpy as np
from math import exp,sqrt,pi
import pandas as pd
import Tesy as sim




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


#tune_SDG();


#simple_SDG_Pa = simple_SDG_Pa()




# # ATTEMPT KERNEL REGRESSION
# class KernelRegression(BaseEstimator, RegressorMixin):
#     def __init__(self, kernel="rbf", gamma=None):
#         self.kernel = kernel
#         self.gamma = gamma
#
#     def fit(self, X, y):
#         self.X = X
#         self.y = y
#
#         if hasattr(self.gamma, "__iter__"):
#             self.gamma = self._tune_gamma(self.gamma)
#
#         return self
#
#     def predict(self, X):
#         K = pairwise_kernels(self.X, X, metric=self.kernel, gamma=self.gamma)
#         return (K * self.y[:, None]).sum(axis=0) / K.sum(axis=0)
#
#     def _tune_gamma(self, gamma_values):
#         # Select specific value of gamma from the range of given gamma_values
#         # by minimizing mean-squared error in leave-one-out cross validation
#         mse = np.empty_like(gamma_values, dtype=np.float)
#         for i, gamma in enumerate(gamma_values):
#             K = pairwise_kernels(self.X, self.X, metric=self.kernel,
#                                  gamma=gamma)
#             np.fill_diagonal(K, 0)  # leave-one-out
#             Ky = K * self.y[:, np.newaxis]
#             y_pred = Ky.sum(axis=0) / K.sum(axis=0)
#             mse[i] = ((y_pred - self.y) ** 2).mean()
#
#         return gamma_values[np.nanargmin(mse)]
#
#     def fit2(test_X, train_X, train_y, bandwidth=1.0, kn='box'):
#         kernels = {
#             'box': lambda x: 1 / 2 if (x <= 1 and x >= -1) else 0,
#             'gs': lambda x: 1 / sqrt(2 * pi) * exp(-x ** 2 / 2),
#             'ep': lambda x: 3 / 4 * (1 - x ** 2) if (x <= 1 and x >= -1) else 0
#         }
#         predict_y = []
#         for entry in test_X:
#             nks = [np.sum((j - entry) ** 2) / bandwidth for j in train_X]
#             ks = [kernels['box'](i) for i in nks]
#             dividend = sum([ks[i] * train_y[i] for i in range(len(ks))])
#             divisor = sum(ks)
#             predict = dividend / divisor
#             predict_y.extend(predict)
#             # print(entry)
#         return np.array(predict_y)[:, np.newaxis]


# ###################################Bishop's Kernel###################
# import numpy as np
# import matplotlib.pyplot as plt
#
# from prml.kernel import PolynomialKernel
# from prml.kernel import RBF
# from prml.kernel import GaussianProcessClassifier
# from prml.kernel import GaussianProcessRegressor
#
#
# def create_toy_data(func, n=10, std=1., domain=[0., 1.]):
#     x = np.linspace(domain[0], domain[1], n)
#     t = func(x) + np.random.normal(scale=std, size=n)
#     return x, t
#
# def sinusoidal(x):
#         return np.sin(2 * np.pi * x)
#
# #TRIAL 1 DUAL REPRESENTATION
# x_train, y_train = create_toy_data(sinusoidal, n=10, std=0.1)
# x = np.linspace(0, 1, 100)
#
# model = GaussianProcessRegressor(kernel=PolynomialKernel(3, 1.), beta=int(1e10))
# model.fit(x_train, y_train)
#
#
# y = model.predict(x)
# plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
# #plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
# plt.plot(x, y, color="r", label="gpr")
# plt.show()
#
#
# # GAUSSIAN PROCESSES
# #TRIAL 2
# x_train, y_train = create_toy_data(sinusoidal, n=7, std=0.1, domain=[0., 0.7])
# x = np.linspace(0, 1, 100)
#
# model = GaussianProcessRegressor(kernel=RBF(np.array([1., 15.])), beta=100)
# model.fit(x_train, y_train)
#
# y, y_std = model.predict(x, with_error=True)
# plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
# #plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
# plt.plot(x, y, color="r", label="gpr")
# #plt.fill_between(x, y - y_std, y + y_std, alpha=0.5, color="pink", label="std")
# plt.show()
# #
#
# #TRIAL 3
# x_train, y_train = create_toy_data(sinusoidal, n=7, std=0.1, domain=[0., 0.7])
# x = np.linspace(0, 1, 100)
#
# print("xtrain", x_train)
# print("ytrain", y_train)
# print("x", x)
#
# plt.figure(figsize=(20, 5))
# plt.subplot(1, 2, 1)
# model = GaussianProcessRegressor(kernel=RBF(np.array([1., 1.])), beta=100)
# model.fit(x_train, y_train)
# y, y_std = model.predict(x, with_error=True)
# plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
# #plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
# plt.plot(x, y, color="r", label="gpr {}".format(model.kernel.params))
# #plt.fill_between(x, y - y_std, y + y_std, alpha=0.5, color="pink", label="std")
# plt.legend()
#
# plt.subplot(1, 2, 2)
# model.fit(x_train, y_train, iter_max=100)
# y, y_std = model.predict(x, with_error=True)
# plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", color="blue", label="training")
# #plt.plot(x, sinusoidal(x), color="g", label="sin$(2\pi x)$")
# plt.plot(x, y, color="r", label="gpr {}".format(np.round(model.kernel.params, 2)))
# #plt.fill_between(x, y - y_std, y + y_std, alpha=0.5, color="pink", label="std")
# plt.legend()
# plt.show()
#
#
# #TRIAL 4
# def create_toy_data_3d(func, n=10, std=1.):
#     x0 = np.linspace(0, 1, n)
#     x1 = x0 + np.random.normal(scale=std, size=n)
#     x2 = np.random.normal(scale=std, size=n)
#     t = func(x0) + np.random.normal(scale=std, size=n)
#     return np.vstack((x0, x1, x2)).T, t
#
# x_train, y_train = create_toy_data_3d(sinusoidal, n=20, std=0.1)
# x0 = np.linspace(0, 1, 100)
# x1 = x0 + np.random.normal(scale=0.1, size=100)
# x2 = np.random.normal(scale=0.1, size=100)
# x = np.vstack((x0, x1, x2)).T
#
# model = GaussianProcessRegressor(kernel=RBF(np.array([1., 1., 1., 1.])), beta=100)
# model.fit(x_train, y_train)
# y, y_std = model.predict(x, with_error=True)
# plt.figure(figsize=(20, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(x_train[:, 0], y_train, facecolor="none", edgecolor="b", label="training")
# plt.plot(x[:, 0], sinusoidal(x[:, 0]), color="g", label="$\sin(2\pi x)$")
# plt.plot(x[:, 0], y, color="r", label="gpr {}".format(model.kernel.params))
# plt.fill_between(x[:, 0], y - y_std, y + y_std, color="pink", alpha=0.5, label="gpr std.")
# plt.legend()
# plt.ylim(-1.5, 1.5)
#
# model.fit(x_train, y_train, iter_max=100, learning_rate=0.001)
# y, y_std = model.predict(x, with_error=True)
# plt.subplot(1, 2, 2)
# plt.scatter(x_train[:, 0], y_train, facecolor="none", edgecolor="b", label="training")
# plt.plot(x[:, 0], sinusoidal(x[:, 0]), color="g", label="$\sin(2\pi x)$")
# plt.plot(x[:, 0], y, color="r", label="gpr {}".format(np.round(model.kernel.params, 2)))
# plt.fill_between(x[:, 0], y - y_std, y + y_std, color="pink", alpha=0.2, label="gpr std.")
# plt.legend()
# plt.ylim(-1.5, 1.5)
# plt.show()


# #TRIAL 5
# def create_toy_data():
#     x0 = np.random.normal(size=50).reshape(-1, 2)
#     x1 = np.random.normal(size=50).reshape(-1, 2) + 2.
#     return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)[:, None]
#
# x_train, y_train = create_toy_data()
# x0, x1 = np.meshgrid(np.linspace(-4, 6, 100), np.linspace(-4, 6, 100))
# x = np.array([x0, x1]).reshape(2, -1).T
#
# model = GaussianProcessClassifier(RBF(np.array([1., 7., 7.])))
# model.fit(x_train, y_train)
# y = model.predict(x)
#
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.ravel())
# plt.contourf(x0, x1, y.reshape(100, 100), levels=np.linspace(0,1,3), alpha=0.2)
# plt.colorbar()
# plt.xlim(-4, 6)
# plt.ylim(-4, 6)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# real trial
# poly1_wall = sim.Polygon(sim.vec2(0,0), sim.vec2(6, -8), "wall")
# poly2_wall = sim.Polygon(sim.vec2(8,0), sim.vec2(14, -8), "wall")
# poly3_wall = sim.Polygon(sim.vec2(16,0), sim.vec2(22, -8), "wall")
#
# poly1_mirr = sim.Polygon(sim.vec2(2,-1), sim.vec2(5, -4), "mirror")
# poly2_mirr = sim.Polygon(sim.vec2(9,-1), sim.vec2(11, -5), "mirror")
# poly3_mirr = sim.Polygon(sim.vec2(17,-1), sim.vec2(21, -3), "mirror")
#
# poly1_sink = sim.Polygon(sim.vec2(2,-5), sim.vec2(5, -6), "sink")
# poly2_sink = sim.Polygon(sim.vec2(9,-6), sim.vec2(11, -7), "sink")
# poly3_sink = sim.Polygon(sim.vec2(17,-4), sim.vec2(21, -5), "sink")
#
# poly_out_sink = sim.Polygon(sim.vec2(25,-6), sim.vec2(29, -7), "sink")
# poly_out_wall = sim.Polygon(sim.vec2(24,0), sim.vec2(30, -8), "wall")
#
# poly_out_mirror1 = sim.Polygon(sim.vec2(24.5,-2), sim.vec2(29.5, -4), "mirror")
# poly_out_mirror2 = sim.Polygon(sim.vec2(25,-3), sim.vec2(29, -5), "mirror")
# poly_out_mirror3 = sim.Polygon(sim.vec2(25,-1), sim.vec2(29, -5), "mirror") #goal

# example1 = sim.Scene("group1")
# example1.addPolygon(poly1_wall)
# example1.addPolygon(poly1_sink)
# example1.addPolygon(poly1_mirr)
#
# example2 = sim.Scene("group2")
# example2.addPolygon(poly2_wall)
# example2.addPolygon(poly2_sink)
# example2.addPolygon(poly2_mirr)
#
# example3 = sim.Scene("group3")
# example3.addPolygon(poly3_wall)
# example3.addPolygon(poly3_sink)
# example3.addPolygon(poly3_mirr)

vecPoly1 = sim.VecPoly()
vecPoly2 = sim.VecPoly()
vecPoly3 = sim.VecPoly()

example1 = sim.Scene("group1")
example1.addPolygon(0.0, 0.0, 6.0, -8.0, "wall", vecPoly1)
example1.addPolygon(2.0, -1.0, 5.0, -4.0, "mirror", vecPoly1)
example1.addPolygon(2.0 ,-5.0, 5.0, -6.0, "sink", vecPoly1)

example2 = sim.Scene("group2")
example2.addPolygon(8.0, 0.0, 14.0, -8.0, "wall", vecPoly2)
example2.addPolygon(9.0, -1.0, 11.0, -5.0, "mirror", vecPoly2)
example2.addPolygon(9.0 ,-6.0, 11.0, -7.0, "sink", vecPoly2)

example3 = sim.Scene("group3")
example3.addPolygon(16.0, 0.0, 22.0, -8.0, "wall", vecPoly3)
example3.addPolygon(17.0, -1.0, 21.0, -3.0, "mirror", vecPoly3)
example3.addPolygon(17.0 ,-4.0, 21.0, -5.0, "sink", vecPoly3)

exampleScenes = [example1, example2, example3];

# output = sim.Scene("output")
# output.addPolygon(poly_out_wall)
# output.addPolygon(poly_out_sink)
vecPolyOut = sim.VecPoly()
output = sim.Scene("output")
output.addPolygon(24.0, 0.0, 30.0, -8.0, "wall", vecPolyOut)
output.addPolygon(25.0,-6.0, 29.0, -7.0, "sink", vecPolyOut)


# print('1_wall_width', poly1_wall.width)
# print('1_wall_height', poly1_wall.height)
# print('2_wall_width', poly2_wall.width)
# print('2_wall_height', poly2_wall.height)
# print('poly3_sink_w', poly3_sink.width)
# print('poly3_sink_h', poly3_sink.height)


# print(vecPoly1[0].low_bound.getX())
# print(vecPoly1[0].low_bound.getY())
# print(vecPoly1[1].label)
# print(vecPoly1[2].label)

vecPolyList = [vecPoly1, vecPoly2, vecPoly3]
allPossibleCandidates = []

desiredLabel = "mirror"
#desiredpolygon in each scene
desiredPolygons = []
# GET CANDIDATE
# For each example scene
for vecPoly in vecPolyList:
    print(vecPoly.size())
    # iterate to find the target desired object
    currPoly = None
    for i in range(vecPoly.size()):
        label = vecPoly[i].label
        print(label)
        if label == desiredLabel:
            currPoly = vecPoly[i]
            desiredPolygons.append(currPoly)
    #print(currPoly.label)
    # iterate through all other polygons to come up with relations
    currminx = currPoly.low_bound.getX()
    currmaxY = currPoly.low_bound.getY()
    currmaxX = currPoly.upper_bound.getX()
    currminy = currPoly.upper_bound.getY()
    lowerBounds = []
    upperBounds = []
    for i in range(vecPoly.size()):
        p = vecPoly[i]
        label = p.label
        if label != desiredLabel:
            minx = p.low_bound.getX()
            maxY = p.low_bound.getY()
            maxX = p.upper_bound.getX()
            miny = p.upper_bound.getY()
            # print(minx, miny, maxX, maxY)
            # print(currminx, currminy, currmaxX, currmaxY)
            newMinX = minx - currminx
            newMinY = miny - currminy
            newMaxX = maxX - currmaxX
            newMaxY = maxY - currmaxY
            for j in range(vecPolyOut.size()):
                if vecPolyOut[j].label == label:
                    outputCurrPoly = vecPolyOut[j]
                    outminx = outputCurrPoly.low_bound.getX()
                    outmaxY = outputCurrPoly.low_bound.getY()
                    outmaxX = outputCurrPoly.upper_bound.getX()
                    outminy = outputCurrPoly.upper_bound.getY()
                    lowerBounds.append([outminx - newMinX, outminy - newMinY])
                    upperBounds.append([outmaxX - newMaxX, outmaxY -newMaxY])
    print(lowerBounds)
    print(upperBounds)
    # Create candiate placements by taking combination of lower and upper
    for lb in lowerBounds:
        for ub in upperBounds:
            allPossibleCandidates.append([lb[0], lb[1], ub[0], ub[1]])

print(allPossibleCandidates)
print(len(allPossibleCandidates))
print(len(desiredPolygons))

# get Polygons from candidate placements
# get feature set x corresponding to each new candidate placement
xList = []
candidatePolygons = []
for c in allPossibleCandidates:
    print(c)
    potentialPoly = sim.Polygon(sim.vec2(c[0], c[1]), sim.vec2(c[2], c[3]), "mirror")
    candidatePolygons.append(potentialPoly)
    #print("AHHHHH", example1.relationship(sim.vec2(2, 3), poly1_mirr))
    xList.append(output.calculateRelationships(potentialPoly))

print("here", xList)
# print(candidatePolygons[0].low_bound.getY())
# print(candidatePolygons[1].low_bound.getY())
# print(candidatePolygons[2].low_bound.getY())
# print(candidatePolygons[3].low_bound.getY())
# print(candidatePolygons[4].low_bound.getY())
# print(candidatePolygons[5].low_bound.getY())
# print(candidatePolygons[6].low_bound.getY())
# print(candidatePolygons[7].low_bound.getY())
# print(candidatePolygons[8].low_bound.getY())
# print(candidatePolygons[9].low_bound.getY())

# get Phi values for each example
phiList = []
for i in range(len(exampleScenes)):
    phiList.append(exampleScenes[i].calculateRelationships(desiredPolygons[i]))

print("phi", phiList)

# Get Similarity Measures
similarityMeasures = sim.SimilarityMeasures()
xphiList = []
for x in xList:
    x1 = []
    for phi in phiList:
        ss = similarityMeasures.shapeSimilarity(x, phi)
        x1.append(ss)
    xphiList.append(x1)

print("xphiList", xphiList)

# get Gram Matrix
GramMatRows = []
for i in phiList:
    currRow = [];
    for j in phiList:
        currRow.append(similarityMeasures.shapeSimilarity(i, j))
    GramMatRows.append(np.asarray(currRow))
GramMat = np.asarray(GramMatRows)

print("GRAM", GramMat)

beta = 1.0
I = np.eye(len(exampleScenes))
covariance = GramMat + I / beta
precision = np.linalg.inv(covariance)

yx = np.matmul(xphiList, precision)
print(yx)

allRates = []
for val in yx:
    allRates.append(val[0] + val[1] + val[2])

maxVal = -np.inf
maxValIndex = 0
for i in range(len(allRates)):
    if (allRates[i] > maxVal):
        maxVal = allRates[i]
        maxValIndex = i

print(maxVal, maxValIndex)
print(allRates)
print(allPossibleCandidates[maxValIndex])

def fit(self, X, t, iter_max=0, learning_rate=0.1):
    """
    maximum likelihood estimation of parameters in kernel function

    Parameters
    ----------
    X : ndarray (sample_size, n_features)
        input
    t : ndarray (sample_size,)
        corresponding target
    iter_max : int
        maximum number of iterations updating hyperparameters
    learning_rate : float
        updation coefficient

    Attributes
    ----------
    covariance : ndarray (sample_size, sample_size)
        variance covariance matrix of gaussian process
    precision : ndarray (sample_size, sample_size)
        precision matrix of gaussian process

    Returns
    -------
    log_likelihood_list : list
        list of log likelihood value at each iteration
    """
    # if X.ndim == 1:
    #     X = X[:, None]
    log_likelihood_list = [-np.Inf]
    self.X = X
    self.t = t
    I = np.eye(len(X))
    Gram = GramMat
    self.covariance = Gram + I / self.beta
    self.precision = np.linalg.inv(self.covariance)
    for i in range(iter_max):
        gradients = self.kernel.derivatives(X, X)
        updates = np.array(
            [-np.trace(self.precision.dot(grad)) + t.dot(self.precision.dot(grad).dot(self.precision).dot(t)) for grad
             in gradients])
        for j in range(iter_max):
            self.kernel.update_parameters(learning_rate * updates)
            Gram = self.kernel(X, X)
            self.covariance = Gram + I / self.beta
            self.precision = np.linalg.inv(self.covariance)
            log_like = self.log_likelihood()
            if log_like > log_likelihood_list[-1]:
                log_likelihood_list.append(log_like)
                break
            else:
                self.kernel.update_parameters(-learning_rate * updates)
                learning_rate *= 0.9
    log_likelihood_list.pop(0)
    return log_likelihood_list


def log_likelihood(self):
    return -0.5 * (
            np.linalg.slogdet(self.covariance)[1]
            + np.matmul(self.t, np.matmul(self.precision, self.t))
            + len(self.t) * np.log(2 * np.pi))


def predict(self, X, with_error=False):
    """
    mean of the gaussian process

    Parameters
    ----------
    X : ndarray (sample_size, n_features)
        input

    Returns
    -------
    mean : ndarray (sample_size,)
        predictions of corresponding inputs
    """
    if X.ndim == 1:
        X = X[:, None]
    K = self.kernel(X, self.X)
    mean = np.matmul(K, np.matmul(self.precision, self.t))
    if with_error:
        var = (
                self.kernel(X, X, False)
                + 1 / self.beta
                - np.sum(np.matmul(K, self.precision) * K, axis=1))
        return mean.ravel(), np.sqrt(var.ravel())
    return mean







# #candidate 1
# # print(poly_out_mirror1.low_bound)
# # print(poly_out_mirror1.label)
# # print("AHHHHH", example1.relationship(sim.vec2(2,3), poly1_mirr))
#
#
# x1 = output.calculateRelationships(poly_out_mirror1)
# x2 = output.calculateRelationships(poly_out_mirror2)
# x3 = output.calculateRelationships(poly_out_mirror3)
# #print("AHIJOFJA:OIFJ",x1)
#
#
# phi1= example1.calculateRelationships(poly1_mirr)
# phi2= example2.calculateRelationships(poly2_mirr)
# phi3= example3.calculateRelationships(poly3_mirr)
# #print("AHIJOFJA:OIFJ",phi1)
#
#
#
#
#
# simMeasures = sim.SimilarityMeasures()
# x11 = simMeasures.shapeSimilarity(x1, phi1)
# x12 = simMeasures.shapeSimilarity(x1, phi2)
# x13 = simMeasures.shapeSimilarity(x1, phi3)
#
# x21 = simMeasures.shapeSimilarity(x2, phi1)
# x22 = simMeasures.shapeSimilarity(x2, phi2)
# x23 = simMeasures.shapeSimilarity(x2, phi3)
#
# x31 = simMeasures.shapeSimilarity(x3, phi1)
# x32 = simMeasures.shapeSimilarity(x3, phi2)
# x33 = simMeasures.shapeSimilarity(x3, phi3)
#
# #x1 = output.calculateRelationshipsVoid()
# kx1 = [x11, x12, x13]
# kx2 = [x21, x22, x23]
# kx3 = [x31, x32, x33]
#
# #GRAM MATRIX K
# K1 = [simMeasures.shapeSimilarity(phi1, phi1), simMeasures.shapeSimilarity(phi1, phi2), simMeasures.shapeSimilarity(phi1, phi3)]
# K2 = [simMeasures.shapeSimilarity(phi2, phi1), simMeasures.shapeSimilarity(phi2, phi2), simMeasures.shapeSimilarity(phi2, phi3)]
# K3 = [simMeasures.shapeSimilarity(phi3, phi1), simMeasures.shapeSimilarity(phi3, phi2), simMeasures.shapeSimilarity(phi3, phi3)]
# KMat = np.asarray([np.asarray(K1), np.asarray(K2), np.asarray(K3)])
# print(KMat)
#
# t = [1, 1, 1]
#
# import matplotlib.pyplot as plt
#
# from prml.kernel import PolynomialKernel
# from prml.kernel import RBF
# from prml.kernel import GaussianProcessClassifier
# from prml.kernel import GaussianProcessRegressor
#
# model = GaussianProcessRegressor(kernel=RBF(np.array([1., 1., 1., 1.])), beta=100)
#
# print(x1.lowLeft.getCornerSize())
#
# print("labels", phi1.label, x1)
# print("relationship x1", x11, x12, x13)
# print("relationship x2", x21, x22, x23)
# print("relationship x3", x31, x32, x33)
#
#
# candidatesVec = [poly_out_mirror1, poly_out_mirror2, poly_out_mirror3]
# similarityMeasuresVec = [x11, x21, x31]
# index = 0
# max = 0.0
# for i in range(3):
#     if (similarityMeasuresVec[i] > max):
#         max = similarityMeasuresVec[i]
#         index = i
#
# bestCandidate = candidatesVec[index]
#
# bestCoord_UpperLeft = [bestCandidate.low_bound.getX(), bestCandidate.low_bound.getY(), 0]
#
# bestCoord_LowerRight = [bestCandidate.upper_bound.getX(), bestCandidate.upper_bound.getY(), 0]
#
# print("Upper_Left", bestCoord_UpperLeft)
# print("Lower_Right", bestCoord_LowerRight)

#
# class KernelReg(object):
#     def __init__(self):
#         self.lowerRight = bestCoord_LowerRight
#         self.upperLeft = bestCoord_UpperLeft
