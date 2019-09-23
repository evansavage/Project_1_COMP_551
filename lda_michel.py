import numpy as np


class Lda2:
    def __init__(self, iter: int, learning_rate: float):
        # dataset = np.asarray(dataset)
        # self.features = dataset[:, :-1]
        # self.labels = dataset[:, -1]
        self.iter = iter
        self.learning_rate = learning_rate

    def fit(self, X: np.array, Y: np.array):
        """ @Params:
            -- X: dataset
            -- Y: labels
            @return:
            -- W: Transformation Matrix"""
        training_labels = np.unique(Y)
        # n_labels = training_labels.shape[0]
        n_features = X.shape[1]
        mean_vectors = []
        scatter_within_matrix = np.zeros((n_features, n_features))
        scatter_between_matrix = np.zeros((n_features, n_features))

        # 1- Compute the mean vector mu
        global_mean = np.mean(X, axis=0)
        # 2-Compute mean vector for each class muc(d dimensions)
        for column in training_labels:
            mean_vectors.append(np.mean(X[Y == column], axis=0))
        # 3- Compute the within class scatter matrix
        # scatter_per_class = sum((x-mu_k)(x-mu_k). T)
        # scatter_within_matrix=  sum (scatter_per_class)
        for column, mean_vector in zip(training_labels, mean_vectors):
            scatter_per_class = np.zeros((n_features, n_features))
            for row in X[Y == column]:
                row, mean_vector = row.reshape(n_features, 1), mean_vector.reshape(n_features, 1)
                scatter_per_class += (row - mean_vector).dot((row - mean_vector).T)
            scatter_within_matrix += scatter_per_class
        # 4- Compute Scatter between-class matrix
        for i, mean_vector in enumerate(mean_vectors):
            class_size = X[Y == i + 1, :].shape[0]
            # Add column Vector
            mean_vector = mean_vector.reshape(n_features, 1)
            global_mean = global_mean.reshape(n_features, 1)
            scatter_between_matrix += class_size * (mean_vector - global_mean).dot((mean_vector - global_mean).T)
        # 5- Compute the eigenvectors and eigenvalues for scatter matrices scatter_within^-1 . scatter_between
        eigen_values, eigen_vectors = np.linalg.eig(
            np.dot(np.linalg.inv(scatter_within_matrix), scatter_between_matrix))
        # 6- Select the EigenVectors of the cooresponding k largest eigenvalues to create d*k matrix w
        eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
        # select 2 largest?
        transform_matrix = np.hstack([eigen_pairs[i][1].reshape(4, 1) for i in range(0, 2)])

        return transform_matrix

    def predict(self, X_new: np.array, W):
        """@Params:
            -- X_new: dataset
            @return:
            -- np array with binary predictions {0,1} for each point"""
        # 7- Use matrix w to transform n*d dataset x into lower n*k dataset y
        predicted_labels = X_new.dot(W)

        return predicted_labels

    # def fit_dummy(self, X:np.array, Y:np.array):
    #     """ dummy function for testing. TODO: remove later once fit is complete returns all 1's in a column"""
    #     return None

    # def predict_dummy(self, X_new:np.array):
    #     """ dummy function for testing. TODO: remove later once predict  is complete returns all 1's in a column"""
    #     return np.ones((X_new.shape[0])).reshape(-1,1)