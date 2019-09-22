
# import files from split project
from dataset_load import load_dataset, clean_dataset_nan
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from cross_validation import evaluate_acc, k_fold_cross_validation


import numpy as np

# load datasets
wine_dataset = load_dataset('winequality-red.csv', ';')
breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data', ',', 1)

# data clearning
wine_dataset = clean_dataset_nan(wine_dataset)
breast_cancer_dataset = clean_dataset_nan(breast_cancer_dataset)[1:]

## testing data
# lr_wine = LogisticRegression(wine_dataset, 100, 0.3)
# lr_cancer = LogisticRegression(breast_cancer_dataset, 1000, 0.01)
# lr_wine.threshold(5)
# lr_wine.normalize()
# lr_wine.show()

# lda_wine = LinearDiscriminantAnalysis(wine_dataset)
# lda_cancer = LinearDiscriminantAnalysis(breast_cancer_dataset)
# print(breast_cancer_dataset)


# print(lr_wine.update_coefficients())
#
#
# lr_cancer.threshold(3)
# print(lr_cancer.update_coefficients())

# lr_cancer.threshold(3)
# lr_cancer.fit()


## TESTING DATA TODO: Comment out and remove before submitting
x = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]])
y = np.array([1,1,103,104,105, 106, 107, 108, 109, 110]).reshape(-1,1)
m = LinearDiscriminantAnalysis()
k_fold_cross_validation(5, x, y, m)
print("DONE")
