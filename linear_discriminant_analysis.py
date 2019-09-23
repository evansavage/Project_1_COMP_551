import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self, iter:int, learning_rate:float):
        # dataset = np.asarray(dataset)
        # self.features = dataset[:, :-1]
        # self.labels = dataset[:, -1]
        self.w = []

    def fit(self, X:np.array, Y:np.array, normalize):
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

        # 1- Compute the total mean vector mu and the mean vector for each class muc(d dimensions)
        # mu = np.mean(X, axis=0).values
        # mu_k = []

        # 1- Compute the mean vector mu
        global_mean = np.mean(X, axis=0)
        # 2-Compute mean vector for each class muc(d dimensions)
        for column in training_labels:
            mean_vectors.append(np.mean(X[Y==column], axis=0))
        # 3- Compute the within class scatter matrix
        # scatter_per_class = sum((x-mu_k)(x-mu_k). T)
        # scatter_within_matrix=  sum (scatter_per_class)
        for column, mean_vector in zip(training_labels, mean_vectors):
            scatter_per_class = np.zeros((n_features, n_features))
            for row in X[Y == column]:
                row, mean_vector = row.reshape(n_features, 1), mean_vector.reshape(n_features, 1)
                scatter_per_class += (row-mean_vector).dot((row-mean_vector).T)
            scatter_within_matrix += scatter_per_class
        # 4- Compute Scatter between-class matrix
        for i, mean_vector in enumerate(mean_vectors):
            class_size = X[Y == i+1, :].shape[0]
            #Add column Vector
            mean_vector = mean_vector.reshape(n_features, 1)
            global_mean = global_mean.reshape(n_features, 1)
            scatter_between_matrix += class_size * (mean_vector - global_mean).dot((mean_vector - global_mean).T)
        # 5- Compute the eigenvectors and eigenvalues for scatter matrices scatter_within^-1 . scatter_between
        eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.inv(scatter_within_matrix), scatter_between_matrix))
        # 6- Select the EigenVectors of the cooresponding k largest eigenvalues to create d*k matrix w
        eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
        #select 2 largest?
        transform_matrix = np.hstack([eigen_pairs[i][1].reshape(4, 1) for i in range(0, 2)])

        return transform_matrix

        # 4- Select the EigenVectors of the cooresponding k largest eigenvalues to create d*k matrix w

        # 5- Use matrix w to transform n*d dataset x into lower n*k dataset y

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

    # def fit_dummy(self, X:np.array, Y:np.array):
    #     """ dummy function for testing. TODO: remove later once fit is complete returns all 1's in a column"""
    #     return None

    # def predict_dummy(self, X_new:np.array):
    #     """ dummy function for testing. TODO: remove later once predict  is complete returns all 1's in a column"""
    #     return np.ones((X_new.shape[0])).reshape(-1,1)
