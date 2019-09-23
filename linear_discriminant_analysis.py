import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        # dataset = np.asarray(dataset)
        # self.features = dataset[:, :-1]
        # self.labels = dataset[:, -1]
        self.w = []

    def fit(self, X:np.array, Y:np.array, normalize=''):
        """ @Params:
            -- X: dataset
            -- Y: labels
            @return:
            -- None (fits internal perameters to model)"""
        self.w = [0.0 for i in range(len(X[0]) + 1)]
        if normalize == 'max':
            X = X / X.max(axis=0)
        elif normalize == 'scale':
            X = X / 10
        py_0 = (Y == 0).sum() / len(Y)
        py_1 = (Y == 1).sum() / len(Y)

        sum_0 = [0.0 for i in range(len(X[0]))]
        sum_1 = [0.0 for i in range(len(X[0]))]
        N_0 = 0
        N_1 = 0
        for i in range(len(Y)):
            if Y[i] == 0:
                sum_0 += X[i]
                N_0 += 1
            elif Y[i] == 1:
                sum_1 += X[i]
                N_1 += 1
        print(sum_0)
        mu_0 = sum_0 / N_0
        mu_1 = sum_1 / N_1
        # print(mu_0, mu_1)
        sigma_0 = 0
        sigma_1 = 0
        for i in range(len(Y)):
            if Y[i] == 0:
                sigma_0 += np.matmul(X[i] - mu_0, np.transpose(X[i] - mu_0))
            elif Y[i] == 1:
                sigma_1 += np.matmul(X[i] - mu_1, np.transpose(X[i] - mu_1))
        sigma = (sigma_0 + sigma_1) / (len(Y) - 2)
        # print(sigma)
        self.w[0] = np.log(py_1/py_0) \
            - 0.5 * np.matmul(np.transpose(mu_1), mu_1) / sigma \
            + 0.5 * np.matmul(np.transpose(mu_0), mu_0) / sigma
        self.w[1:] = (mu_1 - mu_0) / sigma
        print('COEF:', self.w)

<<<<<<< HEAD

        # 4- Select the EigenVectors of the cooresponding k largest eigenvalues to create d*k matrix w

        # 5- Use matrix w to transform n*d dataset x into lower n*k dataset y

=======
>>>>>>> b934b9f6a64bdcb74109fa3910837f9d596308a3
    def predict(self, X:np.array):
        predictions = []
        for row in X:
            predict = self.w[0] + np.matmul(np.transpose(row), self.w[1:])
            if predict > 0:
                predict = 1
            else:
                predict = 0
            predictions.append(predict)
        return np.asarray(predictions).reshape(-1,1)
