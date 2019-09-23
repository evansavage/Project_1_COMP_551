import numpy as np
import matplotlib.pyplot as plt

##############################       PART 1      ###############################

# (1,2) Download and load datasets into numpy objects

def load_dataset(file_name, delimiter, *remove_first_index, **kwargs):
    if remove_first_index:
        dataset = np.genfromtxt(file_name, delimiter=delimiter)[1:]
        dataset = dataset[:, 1:]
    else:
        dataset = np.genfromtxt(file_name, delimiter=delimiter)[1:]
    return dataset

def clean_dataset_nan(dataset):
    return dataset[~np.isnan(dataset).any(axis=1)]

wine_dataset = load_dataset('winequality-red.csv', ';')
breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data', ',')

print('Wine Dataset before cleaning:', wine_dataset.shape)
print('Breast Cancer Dataset before cleaning:', breast_cancer_dataset.shape, '\n')

wine_dataset = clean_dataset_nan(wine_dataset)
breast_cancer_dataset = clean_dataset_nan(breast_cancer_dataset)

print('Wine Dataset after cleaning:', wine_dataset.shape)
print('Breast Cancer Dataset after cleaning:', breast_cancer_dataset.shape)

## (4) Compute some statistics on the data

## Wine quality distribution

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
## Tumor class distribution
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
