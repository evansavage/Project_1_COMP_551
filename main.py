import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# import files from split project
from dataset_load import load_dataset
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from cross_validation import evaluate_acc, k_fold_cross_validation

# load datasets
wine_dataset = load_dataset('winequality-red.csv',
    ';',
    visualize=True,
    # interaction=[['citric acid', 'fixed acidity'], ['density', 'fixed acidity']],
    # remove_columns=['pH', 'residual sugar']
    )
breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data',
    ',',
    # visualize=True,
    # interaction=[['Uniformity of Cell Shape', 'Uniformity of Cell Size']],
    remove_columns=['Sample code number']
    )

X_wine = wine_dataset[:,:-1]
Y_wine = wine_dataset[:, -1].reshape(-1,1)

X_cancer = breast_cancer_dataset[:,:-1]
Y_cancer = breast_cancer_dataset[:,-1].reshape(-1,1)
#
X_wine2 = X_wine.copy()
Y_wine2 = Y_wine.copy()

X_cancer2 = X_cancer.copy()
Y_cancer2 = Y_cancer.copy()

print('*** WINE LR ***')
start = time.time()
wine_LR = LogisticRegression(2500, 0.008)
k_fold_cross_validation(5, X_wine, Y_wine, wine_LR, 5.5, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print('*** CANCER LR ***')
start = time.time()
cancer_LR = LogisticRegression(1200, 0.5, lamda = None)
k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR, 3, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print('*** WINE LR with Ridge Regularization ***')
start = time.time()
wine_LR_reg = LogisticRegression(2500, 0.008, reg='Ridge', lamda = 0.13)
k_fold_cross_validation(5, X_wine, Y_wine, wine_LR_reg, 5.5, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print('*** CANCER LR with Ridge Regularization ***')
start = time.time()
cancer_LR_reg = LogisticRegression(1100, 0.55, reg='Ridge', lamda = 0.01)
k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR_reg, 3, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print('*** WINE LR with Lasso Regularization ***')
start = time.time()
wine_LR_reg = LogisticRegression(2500, 0.009, reg='Lasso', lamda = 0.1)
k_fold_cross_validation(5, X_wine, Y_wine, wine_LR_reg, 5.5, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print('*** CANCER LR with Lasso Regularization ***')
start = time.time()
cancer_LR_reg = LogisticRegression(1100, 0.55, reg='Lasso', lamda = 0.01)
k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR_reg, 3, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print('*** WINE LR with Elastic Regularization ***')
start = time.time()
wine_LR_elastic = LogisticRegression(2500, 0.009, reg='Elastic', lamda = 0.1)
k_fold_cross_validation(5, X_wine, Y_wine, wine_LR_elastic, 5.5, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print('*** CANCER LR with Elastic Regularization ***')
start = time.time()
cancer_LR_elastic = LogisticRegression(1100, 0.55, reg='Elastic', lamda = 0.01)
k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR_elastic, 3, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print('*** WINE LDA ***')
start = time.time()
wine_LDA = LinearDiscriminantAnalysis()
k_fold_cross_validation(5, X_wine2, Y_wine2, wine_LDA, 5.5, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print('*** CANCER LDA ***')
start = time.time()
cancer_LDA = LinearDiscriminantAnalysis()
k_fold_cross_validation(5, X_cancer2, Y_cancer2, cancer_LDA, 3, '', True, False)
end = time.time()
print('Time to complete:',end - start)

print("DONE")
