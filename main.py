from dataset_load import load_dataset, clean_dataset_nan
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import matplotlib.pyplot as plt

wine_dataset = load_dataset('winequality-red.csv', ';')
breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data', ',', 1)

wine_dataset = clean_dataset_nan(wine_dataset)
breast_cancer_dataset = clean_dataset_nan(breast_cancer_dataset)[1:]

lr_wine = LogisticRegression(wine_dataset, 1000, 0.01)
lr_cancer = LogisticRegression(breast_cancer_dataset, 1000, 0.01)

lda_wine = LinearDiscriminantAnalysis(wine_dataset)
lda_cancer = LinearDiscriminantAnalysis(breast_cancer_dataset)
print(breast_cancer_dataset)

lr_wine.threshold(5)
lr_wine.fit()
print(lr_wine.update_coefficients())


lr_cancer.threshold(3)
print(lr_cancer.update_coefficients())

# lr_cancer.threshold(3)
# lr_cancer.fit()

def evaluate_acc( true_lables, target_lables):
  """true labels and target labels should both be numpy column arrays of size n*1"""

  sample_size = data_points.shape[0]



x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]])
y = np.array([101,102,103,104,105, 106, 107, 108, 109, 110]).reshape(-1,1)


def k_fold_cross_validation(k, X, Y, model):
  """Divides dataset X and labels Y into k different bins and runs k-fold cross validation.
  @returns: the average accuracy of the k folds"""

  assert X.shape[0] == Y.shape[0] # ensure that the number of features in X matches Y
  sample_size = X.shape[0]
  fold_size = sample_size // k
  remainder = sample_size % k

  # Calculate the size of each of the k chunks into inclusive sliced ranges e.g. [s0, s1]
  slices = []
  for i in range(k):
    if i == 0:
      slices.append((0,fold_size  + (i < remainder) - 1))
    else:
      slices.append(( slices[i-1][1] + 1, slices[i-1][1] +  fold_size + (i < remainder)))

  total_error = 0

  # Break the data up into chunks for each of the k tests
  #-----------------------------------------------------
  for fold in range(k):
    print("====== Kfold set " + str(fold)+ " =========")
    training_data_1 = x[0 : slices[fold][0], :] # from first example to start of validation set
    training_labels_1 = y[0 : slices[fold][0], :] # "" for labels
    training_data_2 = x[slices[fold][1] + 1:, :] #from end of validation set to end of entire set
    training_labels_2 = y[slices[fold][1] + 1:, :]

    # Concatinate two halves of training data
    training_data = np.vstack((training_data_1, training_data_2))
    training_labels = np.vstack((training_labels_1, training_labels_2))
    # get validaiton data
    validation_data = x[slices[fold][0] : slices[fold][1] + 1, :]
    validation_labels = y[slices[fold][0] : slices[fold][1] + 1, :]

