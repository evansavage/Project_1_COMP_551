import math
import numpy as np


class LogisticRegression(object):
    def __init__(self, dataset):
        dataset = np.asarray(dataset)
        self.dataset = dataset[:, :-1]
        self.labels = dataset[:, -1]
        self.iter = 1000

    def fit(self):
        return

    def predict(self):
        return



def sigmoid(x):
  return 1 / (1 + math.exp(-x))


lr = LogisticRegression([[1,2,3],[4,5,6],[7,8,9]])
print(lr.dataset)
print(lr.labels)
