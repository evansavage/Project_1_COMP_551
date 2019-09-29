import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##############################       PART 1      ###############################

# (1,2) Download and load datasets into numpy objects

def load_dataset(file_name:str, delimiter:str, remove_columns:list=[], visualize=False, thresh=0.5, interaction:list=[]):
    """load dataset from csv file into a numpy array"""
    dataset = pd.read_csv(file_name, delimiter)
    for index, row in dataset.iterrows():
        if '?' in row.values:
            dataset.drop(index, inplace=True)
    dataset = dataset.apply(pd.to_numeric)
    dataset.dropna()
    if remove_columns:
        for column in remove_columns:
            del dataset[column]
    if interaction:
        dataset = add_interaction_terms(dataset, interaction)
    if visualize:
        visualize_dataset(dataset, thresh)

    return dataset.values

def visualize_dataset(dataset, thresh):
    """visualize dataset with seaborn figures"""
    for i, j in enumerate(dataset.columns.values):
        plt.figure(f'{ j } (column { i })')
        class0 = dataset.iloc[:, i] * (dataset.iloc[:, -1] <= thresh)
        class1 = dataset.iloc[:, i] * (dataset.iloc[:, -1] > thresh)
        print(class0.shape, class1.shape)
        sns.distplot(class0[class0 != 0])
        sns.distplot(class1[class1 != 0])

    plt.figure('Heatmap')
    correlation = dataset.corr()
    sns.heatmap(correlation, annot=True)
    plt.show()
    return


def add_interaction_terms(dataset, interaction):
    """adds interaction terms and nonlinear features to the model"""
    new_columns = []
    for i in interaction:
        new_columns.append(dataset[i[0]] * dataset[i[1]])

    for i, column in enumerate(new_columns):
        dataset.insert(0, f'inter{ i }', column)
    return dataset
