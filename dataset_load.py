import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##############################       PART 1      ###############################

# (1,2) Download and load datasets into numpy objects

def load_dataset(file_name:str, delimiter:str, *remove_first_index, **kwargs):
    """load dataset from csv file into a numpy array"""
    if remove_first_index:
        dataset = np.genfromtxt(file_name, delimiter=delimiter)[1:]
        dataset = dataset[:, 1:]
    else:
        dataset = np.genfromtxt(file_name, delimiter=delimiter)[1:]
    return dataset

def clean_dataset_nan(dataset:np.array):
    """Remove any elements which contain NaN or empty values"""
    return dataset[~np.isnan(dataset).any(axis=1)]


def visualize_dataset(file_name:str, delimiter:str, **kwargs):
    names = []
    with open(file_name) as f:
        names = [i.replace('"', '') for i in f.readline().split(delimiter)]
    for key in kwargs:
        if key == 'columns':
            columns = kwargs['columns']
            if columns[-1] !=  len(names)-1:
                 columns += (len(names)-1,)
            print(columns)
            dataset = np.genfromtxt(file_name, delimiter=delimiter, usecols=columns)[1:]
            print(dataset.shape)
            new_names = []
            for i in range(len(names)):
                if i in columns:
                    new_names.append(names[i])
            names = new_names
        else:
            dataset = np.genfromtxt(file_name, delimiter=delimiter)[1:]
    for key in kwargs:
        if key == 'remove_first_index':
            dataset = dataset[:, 1:]
            names = names[1:]

    for i in range(len(names)):
        plt.figure(f'{ names[i] } (column: { i })')
        # plt.hist(dataset[:, i], bins='auto')
        sns.distplot(dataset[:, i])
        print(names[i])
        print(f'Mean: { np.mean(dataset[:, i])}')
        print(f'StdDev: { np.std(dataset[:, i])}')
    plt.show()
    return

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
