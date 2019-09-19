import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self, dataset):
        dataset = np.asarray(dataset)
        self.dataset = dataset[:, :-1]
        self.labels = dataset[:, -1]
        self.iter = 1000

    def fit(self):
        return

    def predict(self):
        return
