from dataset_load import load_dataset, clean_dataset_nan
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import matplotlib.pyplot as plt

wine_dataset = load_dataset('winequality-red.csv', ';')
breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data', ',', 1)

wine_dataset = clean_dataset_nan(wine_dataset)
breast_cancer_dataset = clean_dataset_nan(breast_cancer_dataset)[1:]

lr_wine = LogisticRegression(wine_dataset, 100, 0.3)
lr_cancer = LogisticRegression(breast_cancer_dataset, 1000, 0.01)
lr_wine.threshold(5)
lr_wine.show()

# lda_wine = LinearDiscriminantAnalysis(wine_dataset)
# lda_cancer = LinearDiscriminantAnalysis(breast_cancer_dataset)
# print(breast_cancer_dataset)


print(lr_wine.update_coefficients())
#
#
# lr_cancer.threshold(3)
# print(lr_cancer.update_coefficients())

# lr_cancer.threshold(3)
# lr_cancer.fit()

def evaluate_acc(data_points, true_lables, target_lables):
  # make sure data input in correct types (np.arrays)
  if not isinstance(data_points, np.array):
    raise "Data points must be of type numpy array"
  if not isinstance(true_lables, np.array):
    raise "Data points must be of type numpy array"
  if not isinstance(target_lables, np.array):
    raise "Data points must be of type numpy array"

  count = data_points.shape()[0]





def k_fold_cross_validation(k, dataset, model):

  training_data = 0;
  validation_data = 0;
