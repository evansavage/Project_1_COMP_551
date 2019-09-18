import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##############################       PART 1      ###############################

# (1,2) Download and load datasets into numpy objects

wine_dataset = np.genfromtxt('winequality-red.csv', delimiter=';')[1:]

breast_cancer_dataset = np.genfromtxt('breast-cancer-wisconsin.data', \
    delimiter=',')[1:]

# (3) Clean the data (Unsure how to do yet??)

print('Wine Dataset before cleaning:', wine_dataset.shape)
print('Breast Cancer Dataset before cleaning:', breast_cancer_dataset.shape, '\n')

wine_dataset = wine_dataset[~np.isnan(wine_dataset).any(axis=1)]
breast_cancer_dataset = breast_cancer_dataset[~np.isnan(breast_cancer_dataset).any(axis=1)]

print('Wine Dataset after cleaning:', wine_dataset.shape)
print('Breast Cancer Dataset after cleaning:', breast_cancer_dataset.shape)

# (4) Compute some statistics on the data

# Wine quality distribution

# plt.figure('Wine')
# plt.subplot(2,1,1)
# plt.hist(wine_dataset[:, -1], bins='auto')
# plt.title('Wine Quality Distribution')
# plt.xlabel('Quantitative rating (1-10)')
# plt.ylabel('Count')
# plt.subplot(2,1,2)
# plt.hist(wine_dataset[:, -2], bins='auto')
# plt.title('Alcohol Percentage Distribution')
# plt.xlabel('Percentage')
# plt.ylabel('Count')
# plt.subplots_adjust(hspace=1)
#
# # Tumor class distribution
#
# plt.figure('Breast Cancer')
# plt.subplot(2,1,1)
# plt.hist(breast_cancer_dataset[:, -1], bins='auto')
# plt.title('Tumor Class Distribution')
# plt.xticks([2,4], ['Benign', 'Malignant'])
# plt.xlabel('State of tumor')
# plt.ylabel('Count')
# plt.subplot(2,1,2)
# plt.hist(breast_cancer_dataset[:, -2], bins='auto')
# plt.title('Mitoses Distribution')
# plt.xlabel('Mitoses Amount')
# plt.ylabel('Count')
# plt.subplots_adjust(hspace=1)
#
# plt.show()


##############################       PART 2      ###############################

class LogisticRegression:
    def __init__(self):
        print()

    def fit(self):
        return

    def predict(self):
        return


class LDA:
    def __init__(self):
        print()

    def fit(self):
        return

    def predict(self):
        return
