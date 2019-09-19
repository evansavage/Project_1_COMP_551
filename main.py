from dataset_load import load_dataset, clean_dataset_nan
from logistic_regression import LogisticRegression
from linear_discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import matplotlib.pyplot as plt

wine_dataset = load_dataset('winequality-red.csv', ';')
breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data', ',')

wine_dataset = clean_dataset_nan(wine_dataset)
breast_cancer_dataset = clean_dataset_nan(breast_cancer_dataset)

lr_wine = LogisticRegression(wine_dataset)
lr_cancer = LogisticRegression(breast_cancer_dataset)

lda_wine = LinearDiscriminantAnalysis(wine_dataset)
lda_cancer = LinearDiscriminantAnalysis(breast_cancer_dataset)
