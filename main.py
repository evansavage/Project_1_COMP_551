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
    visualize=True,
    # interaction=[['citric acid', 'fixed acidity'], ['density', 'fixed acidity']],
    remove_columns=['pH', 'residual sugar']
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
##change lamda to change factor (0.01 and 0 are okay)

print('*** WINE LR without Regression ***')
# wine_LR = LogisticRegression(100, 0.2, lamda = None)
# k_fold_cross_validation(5, X_wine, Y_wine, wine_LR, 5.5, '', True, False)

print('*** CANCER LR without Regression ***')
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
wine_LR_reg = LogisticRegression(2500, 0.008, lamda = 0.13)
k_fold_cross_validation(5, X_wine, Y_wine, wine_LR_reg, 5.5, '', True, False)

print('*** CANCER LR with Ridge Regularization ***')
# cancer_LR_reg = LogisticRegression(1200, 0.5,lamda = 0.01)
# k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR_reg, 3, '', True, False)

print('*** WINE LDA ***')
#wine_LDA = LinearDiscriminantAnalysis()
#k_fold_cross_validation(5, X_wine2, Y_wine2, wine_LDA, 5.5, '', True, False)

print('*** CANCER LDA ***')
#cancer_LDA = LinearDiscriminantAnalysis()
#k_fold_cross_validation(5, X_cancer2, Y_cancer2, cancer_LDA, 3, '', True, False)
#
# print("DONE")
