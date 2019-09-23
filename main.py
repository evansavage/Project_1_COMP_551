import numpy as np

# import files from split project
from dataset_load import load_dataset, clean_dataset_nan
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from cross_validation import evaluate_acc, k_fold_cross_validation

# load datasets
wine_dataset = load_dataset('winequality-red.csv', ';')
breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data', ',', 1)

# data clearning
wine_dataset = clean_dataset_nan(wine_dataset)
breast_cancer_dataset = clean_dataset_nan(breast_cancer_dataset)

## TESTING DATA TODO: Comment out and remove before submitting
# x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]])
# y = np.array([1,1,103,104,105, 106, 107, 108, 109, 110]).reshape(-1,1)

X_wine = wine_dataset[:,:-1]
Y_wine = wine_dataset[:, -1].reshape(-1,1)

X_cancer = breast_cancer_dataset[:,:-1]
Y_cancer = breast_cancer_dataset[:,-1].reshape(-1,1)

X_wine2 = wine_dataset[:,:-1]
Y_wine2 = wine_dataset[:, -1].reshape(-1,1)

X_cancer2 = breast_cancer_dataset[:,:-1]
Y_cancer2 = breast_cancer_dataset[:,-1].reshape(-1,1)

wine_LR = LogisticRegression(100, 0.3)
cancer_LR = LogisticRegression(2000, 0.3)

k_fold_cross_validation(5, X_wine, Y_wine, wine_LR, 5, shuffle=123, debug=True)
# k_fold_cross_validation(5, X_cancer, Y_cancer, cancer_LR, 3, shuffle=True, debug=True)

# wine_LDA = LinearDiscriminantAnalysis()
# cancer_LDA = LinearDiscriminantAnalysis()

# k_fold_cross_validation(5, X_wine2, Y_wine2, wine_LDA, 5, '')
# k_fold_cross_validation(5, X_cancer2, Y_cancer2, cancer_LDA, 3, '')

print("DONE")
