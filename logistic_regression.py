import math
import numpy as np


class LogisticRegression(object):
    def __init__(self, iter:int, learning_rate:float, theta, lamda):
        """ Constructor for logistic regression model
        @params:
            -- iter : int = number of iterations
            -- learning_rate : float = how quickly the algorithm converges
            """
        # dataset = np.asarray(features)
        # self.labels = dataset[:,-1]
        # dataset = np.insert(dataset, 0, 1, axis=1)
        # self.dataset = dataset
        self.iter = iter
        self.learning_rate = learning_rate
        self.w = []
        self.theta = theta
        self.lamda = lamda

    def fit(self, X:np.array, Y:np.array, normalize=''):
        # print(Y)
        if normalize == 'max':
            X = X / X.max(axis=0)
        elif normalize == 'scale':
            X = X / 10
        X = np.insert(X, 0, 1, axis=1)
        self.w = [0.0 for i in range(len(X[0]))]
        for _ in range(self.iter):
            #calculate penalty from ridge regression
            num_features = len(Y)
            #Y = Y[:, np.newaxis]
            #penalty calculation according to lamda and theta (ridge regression)
            penalty_gradient = (self.lamda / num_features) * self.theta
            sum = 0
            for j, row in enumerate(X):
                # row = [float(i) for i in row]
                #self.w = self.w @ self.theta[1:]
                sigma = sigmoid(row, self.w)
                sum += row * (Y[j] - sigma)
            # print(sum_error)
            ## Adding penalty to sum for regularization
            new_sum = sum + penalty_gradient

            self.w = self.w @ self.theta + self.learning_rate * new_sum
            self.theta = self.theta - (self.learning_rate * new_sum)
            # print(self.w)

    def predict(self, X:np.array):
        predictions = []
        X = np.insert(X, 0, 1, axis=1)
        for row in X:
            a = np.matmul(np.transpose(self.w), row)
            if a >= 0:
                predictions.append(np.round(1.0 / (1.0 + np.exp(-a))))
            else:
                predictions.append(np.round(np.exp(a) / (1 + np.exp(a))))
        return np.asarray(predictions).reshape(-1,1)

    # def fit_dummy(self, X:np.array, Y:np.array):
    #     """ dummy function for testing. TODO: remove later once fit is complete returns all 1's in a column"""
    #     return None
    #
    # def predict_dummy(self, X_new:np.array):
    #     """ dummy function for testing. TODO: remove later once predict  is complete returns all 1's in a column"""
    #     return np.ones((X_new.shape[0])).reshape(-1,1)


def sigmoid(x, w:np.matrix):
    # print(x, w)
    a = np.matmul(np.transpose(w), x)
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1 + math.exp(a))


# lr = LogisticRegression([[1,2,3],[4,5,6],[7,8,9]])
# print(lr.dataset)
# print(lr.labels)
