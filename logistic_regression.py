import math
import numpy as np


class LogisticRegression(object):
    def __init__(self, iter:int, learning_rate:float, reg:string=None, lamda=None):
        """ Constructor for logistic regression model
        @params:
            -- iter : int = number of iterations
            -- learning_rate : float = how quickly the algorithm converges
            -- reg. regularization type ['Elastic' | 'Ridge' | 'Lasso']
            """
        # dataset = np.asarray(features)
        # self.labels = dataset[:,-1]
        # dataset = np.insert(dataset, 0, 1, axis=1)
        # self.dataset = dataset
        self.iter = iter
        self.learning_rate = learning_rate
        self.w = []
        self.lamda = lamda
        self.reg = reg
        if self.reg:
            print("Iter:", self.iter, "| LR:", self.learning_rate, "| Reg:", self.reg, "| Lamda:", self.lamda)
        else:
            print("Iter:", self.iter, "| LR:", self.learning_rate)

    def fit(self, X:np.array, Y:np.array, normalize=''):

        self.costs = []
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
                # row = [float(i) for i in row]
                sigma = sigmoid(row, self.w)
                sum += row * (Y[j] - sigma)
            # print(sum_error)
            #Ridge Regression Regularization
            if self.reg:
                if self.reg == 'Ridge':
                    if self.lamda is not None:
                        # self.w = self.learning_rate * (self.lamda * sum + np.sum(self.w))
                        self.w = self.w + self.learning_rate * (sum + np.multiply(2 * self.lamda, self.w))
                    else:
                        self.w = self.w + self.learning_rate * sum
                elif self.reg == 'Lasso':
                    if self.lamda is not None:
                        # self.w = self.learning_rate * (self.lamda * sum + np.sum(self.w))
                        self.w = self.w + self.learning_rate * (sum + np.multiply(self.lamda, np.sign(self.w)))
                    else:
                        self.w = self.w + self.learning_rate * sum
                elif self.reg == 'Elastic':
                    if self.lamda is not None:
                        # self.w = self.learning_rate * (self.lamda * sum + np.sum(self.w))
                        self.w = self.w + self.learning_rate * (sum + np.multiply(self.lamda, np.sign(self.w)) + np.multiply(2 * self.lamda, self.w))
                    else:
                        self.w = self.w + self.learning_rate * sum
            else:
                self.w = self.w + self.learning_rate * sum

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
