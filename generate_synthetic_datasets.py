
from __future__ import division
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd

from util.functions import generate_multivariate_gaussians


parameters_synthetic = {

            'dataset1': {"gaussians": np.array([
                [99, 9, 99, 169, 200, 0],
                [108, 9, 99, 169, 200, 1]
            ])},
            'dataset2':   {"gaussians": np.array([
                [99, 9, 99, 169, 200, 0],
                [104, 16, 138, 16, 150, 0],
                [104, 16, 60, 16, 150, 1],
                [108, 9, 99, 169, 200, 1]
            ])},
            'dataset3': {"gaussians": np.array([
                        [99, 9, 99, 169, 44, 25, 200, 0],
                        [104, 16, 138, 16, 44, 25, 150, 0],
                        [104, 16, 60, 16, 41, 25, 150, 1],
                        [108, 9, 99, 169, 41, 25, 200, 1]
            ])},
            'dataset4': { "gaussians": np.array([
                        [99, 9, 99, 169, 200, 0],
                        [104, 16, 118, 16, 150, 0],
                        [104, 16, 80, 16, 150, 1],
                        [100, 9, 99, 169, 200, 1]
                    ])}}



np.random.seed(1)

for dataset in ['dataset1','dataset2','dataset3','dataset4']:

    data =  generate_multivariate_gaussians(parameters_synthetic[dataset]["gaussians"], 10)
    X, y = data.data, data.target

    if dataset == 'dataset3':
        df = pd.DataFrame(data=X,columns=["x1_min","x1_max","x2_min","x2_max","x3_min","x3_max"])
        df['target'] = y
    else:
        df = pd.DataFrame(data=X,columns=["x1_min","x1_max","x2_min","x2_max"])
        df['target'] = y
    df.to_csv("synthetic_datasets\synthetic-{}.csv".format(dataset), index=False)
