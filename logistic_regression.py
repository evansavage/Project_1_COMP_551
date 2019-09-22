import math
import numpy as np


class LogisticRegression(object):
    def __init__(self, dataset, iter, learning_rate):
        # dataset = np.asarray(features)
        # self.labels = dataset[:,-1]
        dataset = np.insert(dataset, 0, 1, axis=1)
        self.dataset = dataset
        self.iter = iter
        self.learning_rate = learning_rate

    def threshold(self, value):
        for row in self.dataset:
            if row[-1] > value:
                row[-1] = 1
            else:
                row[-1] = 0

    def normalize(self):
        self.dataset = self.dataset / self.dataset.max(axis=0)

    def fit(self):
        N = len(self.labels)
        for i in range(0, N-1):
            print(self.labels[i])
    def show(self):
        print(self.dataset)
        # print(self.labels[200:300])
    def check_temp(self, w):
        for row in self.dataset:
            print(sigmoid(row[:-1], w))

    def update_coefficients(self):
        w = [0.0 for i in range(len(self.dataset[0])-1)]
        for i in range(self.iter):
            sum = 0
            for row in self.dataset:
                sigma = sigmoid(row[:-1], w)
                sum += row[:-1] * (row[-1] - sigma)
            # print(sum_error)
            w = w + self.learning_rate*sum
            print(w)
        return w


    def predict(self):
        # for row in
        return



def sigmoid(x, w):
    a = np.matmul(np.transpose(w),x)
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1 + math.exp(a))


# lr = LogisticRegression([[1,2,3],[4,5,6],[7,8,9]])
# print(lr.dataset)
# print(lr.labels)
