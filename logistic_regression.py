import math
import numpy as np

class LogisticRegression(object):
    def __init__(self, iter:int, learning_rate:float, reg:str=None, lamda:float=None):
        """ Constructor for logistic regression model
        @params:
            -- iter : int = number of iterations
            -- learning_rate : float = how quickly the algorithm converges
            -- reg : regularization type ['Elastic' | 'Ridge' | 'Lasso']
            -- lamda : regularization coefficient
            """
        self.iter = iter
        self.learning_rate = learning_rate
        self.w = []
        self.lamda = lamda
        self.reg = reg
        self.trackw = False
        if self.reg:
            print("Iter:", self.iter, "| LR:", self.learning_rate, "| Reg:", self.reg, "| Lamda:", self.lamda)
        else:
            print("Iter:", self.iter, "| LR:", self.learning_rate)

    def fit(self, X:np.array, Y:np.array, normalize:str=''):
        """
        Fit logisitic regression / gradient descent model
        @params:
            -- X: training examples without label
            -- Y: training labels
            -- normalize: string name of normalizing function ['' (none), 'max', 'scale']
        """
        self.costs = []
        self.ws = [] # list of training errors to determine convergence speed

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
            #Ridge Regression Regularization
            if self.trackw:
                self.ws.append(self.w)
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
        """
        Predict output using already trained model
        @params:
            -- X: input data for prediciton
        """
        predictions = []
        X = np.insert(X, 0, 1, axis=1)
        for row in X:
            a = np.matmul(np.transpose(self.w), row)
            if a >= 0:
                predictions.append(np.round(1.0 / (1.0 + np.exp(-a))))
            else:
                predictions.append(np.round(np.exp(a) / (1 + np.exp(a))))
        return np.asarray(predictions).reshape(-1,1)

def sigmoid(x, w:np.matrix):
    """
    Numerically stable sigmoid function.
    """
    a = np.matmul(np.transpose(w), x)
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1 + math.exp(a))
