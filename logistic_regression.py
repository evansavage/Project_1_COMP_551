import math
import numpy as np


class LogisticRegression(object):
    def __init__(self, dataset, iter, learning_rate):
        dataset = np.asarray(dataset)
        self.features = dataset[:, :-1]
        self.labels = dataset[:, -1]
        self.iter = iter
        self.learning_rate = learning_rate

    def threshold(self, value):
        for i in range(0, len(self.labels)-1):
            if self.labels[i] > value:
                self.labels[i] = 1
            else:
                self.labels[i] = 0

    def fit(self):
        N = len(self.labels)
        for i in range(0, N-1):
            print(self.labels[i])

    def update_coefficients(self):
        w = [0 for i in range(len(self.features[0]) + 1)]
        for i in range(self.iter):
            for j, row in enumerate(self.features):
                sigma = sigmoid(row, w)
                error = self.labels[j] - sigma
                w[0] = w[0] + self.learning_rate * error * sigma * (1 - sigma)
                for k in range(1, len(row)):
                    w[k] = w[k] + self.learning_rate * error * sigma * (1 - sigma) * row[k-1]

        return w

    def predict(self):
        return



def sigmoid(x, w):
    a = w[0]
    for i in range(len(x)):
        a += w[i+1]*x[i]
    return 1 / (1 + math.exp(-a))


# lr = LogisticRegression([[1,2,3],[4,5,6],[7,8,9]])
# print(lr.dataset)
# print(lr.labels)
