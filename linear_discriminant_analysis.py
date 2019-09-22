import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        # dataset = np.asarray(dataset)
        # self.features = dataset[:, :-1]
        # self.labels = dataset[:, -1]
        self.iter = 1000
        self.learning_rate = 0.01

    def fit(self, X:np.array, Y:np.array):
        """ @Params:
            -- X: dataset
            -- Y: labels
            @return:
            -- None (fits internal perameters to model)"""
        # 1- Compute the total mean vector mu and the mean vector for each class muc(d dimensions)
        mu = np.mean(X, axis=0).values
        mu_k = []

        # 2- Compute the scatter matrices (in between and within class)
        # within_class_scatter=  sum (scatter_per_class)
        # scatter_per_class = sum((x-mi)(x-mi). T)
        # 3- Compute the eigenvectors and eigenvalues for scatter matrices

        # 4- Select the EigenVectors of the cooresponding k largest eigenvalues to create d*k matrix w

        # 5- Use matrix w to transform n*d dataset x into lower n*k dataset y
        return None

    def predict(self, X_new:np.array):
        """@Params:
            -- X_new: dataset
            @return:
            -- np array with binary predictions {0,1} for each point"""
        return None

    def fit_dummy(self, X:np.array, Y:np.array):
        """ dummy function for testing. TODO: remove later once fit is complete returns all 1's in a column"""
        return None

    def predict_dummy(self, X_new:np.array):
        """ dummy function for testing. TODO: remove later once predict  is complete returns all 1's in a column"""
        return np.ones((X_new.shape[0])).reshape(-1,1)
