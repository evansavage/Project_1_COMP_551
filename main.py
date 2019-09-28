import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import files from split project
from dataset_load import load_dataset
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from cross_validation import evaluate_acc, k_fold_cross_validation



# load datasets
wine_dataset = load_dataset('winequality-red.csv',
    ';',
    # visualize=True,
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
# wine_LR = LogisticRegression(2500, 0.008)
# k_fold_cross_validation(5, X_wine, Y_wine, wine_LR, 5.5, '', True, False)

print('*** CANCER LR ***')
# cancer_LR = LogisticRegression(1200, 0.5, lamda = None)
# k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR, 3, '', True, False)


def chris():
  for i in range(X_wine.shape[1]):
    plt.figure()
    plt.scatter(X_wine[:,i], Y_wine)
    plt.savefig(f"graphs/fig_{i}.png")

def chris2():
  fig1, f1_axes = plt.subplots(ncols=11,nrows=11)
  fig1.set_size_inches(30,30)

  for i in range(11):
    for j in range(11):
      f1_axes[i][j].scatter(X_wine[:,i]*X_wine[:,j], Y_wine)

  fig1.savefig("graphs/wine_interaction.png")

def chris3():
  fig1, f1_axes = plt.subplots(ncols=9,nrows=9)
  fig1.set_size_inches(30,30)

  for i in range(9):
    for j in range(9):
      f1_axes[i][j].scatter(X_cancer[:,i]*X_cancer[:,j], Y_cancer)

  fig1.savefig("graphs/cancer_interaction.png")

print('*** WINE LR with Ridge Regularization ***')
# wine_LR_reg = LogisticRegression(2500, 0.008, reg='Ridge', lamda = 0.13)
# k_fold_cross_validation(5, X_wine, Y_wine, wine_LR_reg, 5.5, '', True, False)

print('*** CANCER LR with Ridge Regularization ***')
# cancer_LR_reg = LogisticRegression(1200, 0.5, reg='Ridge', lamda = 0.001)
# k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR_reg, 3, '', True, False)

print('*** WINE LR with Lasso Regularization ***')
# wine_LR_reg = LogisticRegression(1500, 0.2, reg='Lasso', lamda = 0.01)
# k_fold_cross_validation(5, X_wine, Y_wine, wine_LR_reg, 5.5, '', True, False)

print('*** CANCER LR with Lasso Regularization ***')
# cancer_LR_reg = LogisticRegression(1200, 0.5, reg='Lasso', lamda = 0.01)
# k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR_reg, 3, '', True, False)

print('*** WINE LR with Elastic Regularization ***')
# wine_LR_elastic = LogisticRegression(2500, 0.008, reg='Elastic', lamda = 0.13)
# k_fold_cross_validation(5, X_wine, Y_wine, wine_LR_elastic, 5.5, '', True, False)

print('*** CANCER LR with Elastic Regularization ***')
cancer_LR_elastic = LogisticRegression(1200, 0.5, reg='Elastic', lamda = 0.01)
k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR_elastic, 3, '', True, False)

print('*** WINE LDA ***')
wine_LDA = LinearDiscriminantAnalysis()
k_fold_cross_validation(5, X_wine2, Y_wine2, wine_LDA, 5.5, '', True, False)

print('*** CANCER LDA ***')
cancer_LDA = LinearDiscriminantAnalysis()
k_fold_cross_validation(5, X_cancer2, Y_cancer2, cancer_LDA, 3, '', True, False)

print("DONE")
