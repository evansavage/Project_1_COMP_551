import dataset_load
import logistic_regression
import linear_discriminant_analysis


import numpy as np
import matplotlib.pyplot as plt

def evaluate_acc(data_points, true_lables, target_lables):
  if not isinstance(data_points, np.array):
    raise "Data points must be of type numpy array"
  if not isinstance(true_lables, np.array):
    raise "Data points must be of type numpy array"
  if not isinstance(target_lables, np.array):
    raise "Data points must be of type numpy array"

