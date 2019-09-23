import math
import numpy as np


class LogisticRegression(object):
    def __init__(self, iter:int, learning_rate:float):
        # dataset = np.asarray(features)
        # self.labels = dataset[:,-1]
        # dataset = np.insert(dataset, 0, 1, axis=1)
        # self.dataset = dataset
        self.iter = iter
        self.learning_rate = learning_rate
        self.w = []

    def fit(self, X:np.array, Y:np.array, normalize):
        # print(Y)
        if normalize == 'max':
            X = X / X.max(axis=0)
        elif normalize == 'scale':
            X = X / 10
        X = np.insert(X, 0, 1, axis=1)
        self.w = [0.0 for i in range(len(X[0]))]
        for _ in range(self.iter):
            sum = 0
            for j, row in enumerate(X):
                sigma = sigmoid(row, self.w)
                sum += row * (Y[j] - sigma)
            # print(sum_error)
            self.w = self.w + self.learning_rate*sum
            # print(self.w)

    def predict(self, X:np.array):
        predictions = []
        X = np.insert(X, 0, 1, axis=1)
        for row in X:
            a = np.matmul(np.transpose(self.w),row)
            if a >= 0:
                predictions.append(1.0 / (1.0 + math.exp(-a)))
            else:
                predictions.append(math.exp(a) / (1 + math.exp(a)))
        return np.asarray(predictions).reshape(-1,1)

    # def fit_dummy(self, X:np.array, Y:np.array):
    #     """ dummy function for testing. TODO: remove later once fit is complete returns all 1's in a column"""
    #     return None
    #
    # def predict_dummy(self, X_new:np.array):
    #     """ dummy function for testing. TODO: remove later once predict  is complete returns all 1's in a column"""
    #     return np.ones((X_new.shape[0])).reshape(-1,1)


def sigmoid(x:np.matrix, w:np.matrix):
    a = np.matmul(np.transpose(w),x)
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1 + math.exp(a))


# lr = LogisticRegression([[1,2,3],[4,5,6],[7,8,9]])
# print(lr.dataset)
# print(lr.labels)
