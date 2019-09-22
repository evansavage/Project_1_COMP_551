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

def evaluate_acc(true_labels, predicted_labels):
  """Calculate accuracy of a prediction against the true values
  @Params:
    -- true_labels: the true values of the labels
    -- predicted_labels: the labels from a model prediction
  @return: accuracy as a float from 0 to 1. i.e. # right / # input"""
  if (true_labels.shape != predicted_labels.shape):
    raise Exception("Lables must have same dimensions. n x 1")

  total_entries = true_labels.shape[0]
  accuracy_matrix = true_labels == predicted_labels # returns a column vec. with 1 if corresponding entries are equal else 0
  correct_count = np.sum(accuracy_matrix)

  return correct_count/total_entries

## TESTING DATA
x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]])
y = np.array([101,102,103,104,105, 106, 107, 108, 109, 110]).reshape(-1,1)


def k_fold_cross_validation(k, X, Y, model):
  """Divides dataset X and labels Y into k different bins and runs k-fold cross validation.
  @Params:
    -- k: the number of folds (usually 5)
    -- X: training + validation data combined set
    -- Y: training + validation labels combined set
    -- model: the pre-initiated model used for training NOTE: set of model (e.g. learning rate) before running
  @returns: the average accuracy of the k folds"""

  if k<=0 or type(k) != 'int':
    raise Exception("k must be an integer greater than 0")
  if not isinstance (model, (LogisticRegression, LinearDiscriminantAnalysis)):
    raise Exception("Model must be an instance of either LDA or Log. regression class")
  if X.shape[0] != Y.shape[0]:
    raise Exception("data set (X) must have same number of examples as training set (Y)")
  if Y.shape != (1,):
    raise Exception("label set (Y) must be column vector")

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
    print(f"====== Kfold set #{fold} =========")
    training_data_1 = X[0 : slices[fold][0], :] # from first example to start of validation set
    training_labels_1 = Y[0 : slices[fold][0], :] # "" for labels
    training_data_2 = X[slices[fold][1] + 1:, :] #from end of validation set to end of entire set
    training_labels_2 = Y[slices[fold][1] + 1:, :]

    # Concatinate two halves of training data
    training_data = np.vstack((training_data_1, training_data_2))
    training_labels = np.vstack((training_labels_1, training_labels_2))
    # get validaiton data
    validation_data = X[slices[fold][0] : slices[fold][1] + 1, :]
    validation_labels = Y[slices[fold][0] : slices[fold][1] + 1, :]

    model.fit(training_data, training_labels)
    predicted_labels = model.predict(validation_data)

    accuracy = evaluate_acc(validation_labels, predicted_labels)
    print(f"... fold #{fold} finished with accuracy: {accuracy}")
    total_error += accuracy

  # average error over k entries
  average_error = total_error / k
  print(f"===================== \nFINISHED KFOLD. Average model accuracy: {average_error}")
  return average_error