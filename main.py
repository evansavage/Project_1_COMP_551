import numpy as np
import matplotlib.pyplot as plt

# import files from split project
from dataset_load import load_dataset, clean_dataset_nan, visualize_dataset
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from cross_validation import evaluate_acc, k_fold_cross_validation



# load datasets
wine_dataset = load_dataset('winequality-red.csv', ';')
breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data', ',', [1])

wine_headers = [""]

# data clearning
wine_dataset = clean_dataset_nan(wine_dataset)
breast_cancer_dataset = clean_dataset_nan(breast_cancer_dataset)

# visualize_dataset('winequality-red.csv', ';', columns=(0,1,2,6,5))
# visualize_dataset('breast-cancer-wisconsin.data', ',',columns=(1,2))

## TESTING DATA TODO: Comment out and remove before submitting
# x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]])
# y = np.array([1,1,103,104,105, 106, 107, 108, 109, 110]).reshape(-1,1)

X_wine = wine_dataset[:,:-1]
Y_wine = wine_dataset[:, -1].reshape(-1,1)

X_cancer = breast_cancer_dataset[:,:-1]
Y_cancer = breast_cancer_dataset[:,-1].reshape(-1,1)

X_wine2 = X_wine.copy()
Y_wine2 = Y_wine.copy()

X_cancer2 = X_cancer.copy()
Y_cancer2 = Y_cancer.copy()

wine_LR = LogisticRegression(100, 0.3)
cancer_LR = LogisticRegression(100, 0.3)

# k_fold_cross_validation(5, X_wine, Y_wine, wine_LR, 5.5, '', True, False)
# k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR, 3, '', True, False)
#
# wine_LDA = LinearDiscriminantAnalysis()
# cancer_LDA = LinearDiscriminantAnalysis()
#
# k_fold_cross_validation(5, X_wine2, Y_wine2, wine_LDA, 5, '', True, False)
# k_fold_cross_validation(5, X_cancer2, Y_cancer2, cancer_LDA, 3, '', True, False)

print("DONE")


def chris():
  for i in range(X_wine.shape[1]):
    plt.figure()
    plt.scatter(X_wine[:,i], Y_wine)
    plt.savefig(f"graphs/fig_{i}.png")


# import matplotlib.backends.backend_pdf
import matplotlib.gridspec as gridspec


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

