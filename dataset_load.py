import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##############################       PART 1      ###############################

# (1,2) Download and load datasets into numpy objects

def load_dataset(file_name:str, delimiter:str, remove_columns:list=[], visualize=False, interaction:list=[]):
    """load dataset from csv file into a numpy array"""
    dataset = pd.read_csv(file_name, delimiter)
    if remove_columns:
        for column in remove_columns:
            del dataset[column]
    # print(dataset.shape)
    for index, row in dataset.iterrows():
        if '?' in row.values:
            dataset.drop(index, inplace=True)
    # print(dataset.shape)
    if interaction:
        dataset = add_interaction_terms(dataset, interaction)
    if visualize:
        visualize_dataset(dataset)
    dataset = dataset.apply(pd.to_numeric)
    dataset.dropna()
    # print(dataset.dtypes)
    return dataset.values

# def clean_dataset_nan(dataset:np.array):
#     """Remove any elements which contain NaN or empty values"""
#     return dataset[~np.isnan(dataset).any(axis=1)]




def visualize_dataset(dataset):
    for i, j in enumerate(dataset.columns.values):
        print(i)
        plt.figure(f'{ j } (column { i })')
        class0 = dataset.iloc[:, i] * (dataset.iloc[:, -1] <= 5)
        class1 = dataset.iloc[:, i] * (dataset.iloc[:, -1] > 5)
        sns.distplot(class0[class0 != 0])
        sns.distplot(class1[class1 != 0])

    plt.figure('Heatmap')
    correlation = dataset.corr()
    sns.heatmap(correlation, annot=True)
    plt.show()
    return


def getWineHeaderNames():
  return np.array([
    "fixed acidity",
    "vliatile acidity",
    "citric acid",
    "sugar",
    "chlorides",
    "free SO2",
    "total SO2",
    "density",
    "pH",
    "sulphates",
    "alcohol"
  ])

def getCancerHeaderNames():
  return np.array([
    "clump thickness",
    "cell size uniformity",
    "cell shape uniformity",
    "marginal adhesionn",
    "single epithelial cell size",
    "bare nuclei",
    "bland chromatin",
    "normal nucleoli",
    "mitoses"
  ])

def add_interaction_terms(dataset, interaction):
    new_columns = []
    for i in interaction:
        new_columns.append(dataset[i[0]] * dataset[i[1]])

    for i, column in enumerate(new_columns):
        dataset.insert(0, f'inter{ i }', column)
    return dataset

# wine_dataset = load_dataset('winequality-red.csv', ';')
# breast_cancer_dataset = load_dataset('breast-cancer-wisconsin.data', ',')

# print('Wine Dataset before cleaning:', wine_dataset.shape)
# print('Breast Cancer Dataset before cleaning:', breast_cancer_dataset.shape, '\n')

# wine_dataset = clean_dataset_nan(wine_dataset)
# breast_cancer_dataset = clean_dataset_nan(breast_cancer_dataset)

# print('Wine Dataset after cleaning:', wine_dataset.shape)
# print('Breast Cancer Dataset after cleaning:', breast_cancer_dataset.shape)

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
