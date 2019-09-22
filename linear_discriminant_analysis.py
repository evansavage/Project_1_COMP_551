import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        # dataset = np.asarray(dataset)
        # self.features = dataset[:, :-1]
        # self.labels = dataset[:, -1]
        self.iter = 1000
        self.learning_rate = 0.01

    def fit(self, X, Y):
        """ @Params:
            -- X: dataset
            -- Y: labels
            @return:
            -- None (fits internal perameters to model)"""

        return None

    def predict(self, X_new):
        """@Params:
            -- X_new: dataset
            @return:
            -- np array with binary predictions {0,1} for each point"""

        return np.ones((X_new.shape[0]))

    def setLearningRate(self, learning_rate):
        self.learning_rate = learning_rate

    def setIter(self, iters):
        self.iter = iters