import numpy as np
import pandas as pd

# import files from split project
from dataset_load import load_dataset
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from cross_validation import evaluate_acc, k_fold_cross_validation

# load datasets
wine_dataset = load_dataset('winequality-red.csv',
    ';',
    # interaction=[['citric acid', 'fixed acidity'], ['density', 'fixed acidity']]
    )
breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data',
    ',',
    # visualize=True,
    # interaction=[['Uniformity of Cell Shape', 'Uniformity of Cell Size']],
    remove_columns=['Sample code number'])

# wine_dataset_test = pd.read_csv('winequality-red.csv', ';')
# print(wine_dataset_test.iloc[:, -1])
# breast_cancer_dataset_test = pd.read_csv('breast-cancer-wisconsin.data', ',')
# print(breast_cancer_dataset_test.iloc[:,-1])
# data clearning
# wine_dataset = clean_dataset_nan(wine_dataset)
# breast_cancer_dataset = clean_dataset_nan(breast_cancer_dataset)

## TESTING DATA TODO: Comment out and remove before submitting
# x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]])
# y = np.array([1,1,103,104,105, 106, 107, 108, 109, 110]).reshape(-1,1)

X_wine = wine_dataset[:,:-1]
Y_wine = wine_dataset[:, -1].reshape(-1,1)

X_cancer = breast_cancer_dataset[:,:-1]
Y_cancer = breast_cancer_dataset[:,-1].reshape(-1,1)
#
X_wine2 = X_wine.copy()
Y_wine2 = Y_wine.copy()

X_cancer2 = X_cancer.copy()
Y_cancer2 = Y_cancer.copy()
#
initial_theta_wine = np.zeros((wine_dataset.shape[1], 1))
#
wine_LR = LogisticRegression(100, 0.2, lamda = None)
cancer_LR = LogisticRegression(1200, 0.5, lamda = None)
##change lamda to change factor (0.01 and 0 are okay)
wine_LR_reg = LogisticRegression(100, 0.2, lamda = 0)
cancer_LR_reg = LogisticRegression(1200, 0.5,lamda = 0)
#
print('*** WINE LR without Regression ***')
k_fold_cross_validation(5, X_wine, Y_wine, wine_LR, 5.5, '', True, False)
print('*** CANCER LR without Regression ***')
k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR, 3, '', True, False)
print('*** WINE LR with Ridge Regularization ***')
k_fold_cross_validation(5, X_wine, Y_wine, wine_LR_reg, 5.5, '', True, False)
print('*** CANCER LR with Ridge Regularization ***')
k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR_reg, 3, '', True, False)
# #
#wine_LDA = LinearDiscriminantAnalysis()
#cancer_LDA = LinearDiscriminantAnalysis()
#
print('*** WINE LDA ***')
#k_fold_cross_validation(5, X_wine2, Y_wine2, wine_LDA, 5.5, '', True, False)
print('*** CANCER LDA ***')
#k_fold_cross_validation(5, X_cancer2, Y_cancer2, cancer_LDA, 3, '', True, False)
#
# print("DONE")
