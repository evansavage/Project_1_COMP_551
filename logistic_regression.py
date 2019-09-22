import math
import numpy as np


class LogisticRegression(object):
    def __init__(self, dataset, iters, learning_rate):
        # dataset = np.asarray(features)
        # self.labels = dataset[:,-1]
        self.dataset = dataset
        self.iters = iters
        self.learning_rate = learning_rate

    def threshold(self, value):
        for row in self.dataset:
            if row[-1] > value:
                row[-1] = 1
            else:
                row[-1] = 0

    def fit(self):
        N = len(self.labels)
        for i in range(0, N-1):
            print(self.labels[i])
    def show(self):
        print(self.dataset)
        # print(self.labels[200:300])

    def update_coefficients(self):
        w = [0.0 for i in range(len(self.dataset[0]))]
        for i in range(self.iter):
            sum_error = 0
            for row in self.dataset:
                sigma = sigmoid(row, w)
                # print(sigma, row[-1])
                error = row[-1] - sigma
                sum_error += error**2
                # print(error)
                w[0] = w[0] + self.learning_rate * error * sigma * (1.0 - sigma)
                for k in range(len(row)-1):
                    w[k+1] = w[k+1] + self.learning_rate * error * sigma * (1.0 - sigma) * row[k]
            print(sum_error)
        return w


    def predict(self):
        # for row in
        return



def sigmoid(x, w):
    a = w[0]
    for i in range(len(x) - 1):
        a += w[i+1]*x[i]
    return 1.0 / (1.0 + math.exp(-a))


# lr = LogisticRegression([[1,2,3],[4,5,6],[7,8,9]])
# print(lr.dataset)
# print(lr.labels)
