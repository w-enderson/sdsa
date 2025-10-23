from __future__ import division
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from util.functions import generate_multivariate_gaussians
from models.sdsa import SDSA
import argparse
import time
import os

# Parallelization
import itertools
from multiprocessing import cpu_count, Pool

from models.ivabc import IVABC


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from models.knn import IntervalKNN


classifiers = {
      'ivabc': IVABC,
      'knn': IntervalKNN,
      'sdsa' : SDSA,
      'sdsa_not_update' : SDSA,
      'sdsa_rf' : SDSA,
      'sdsa_rf_not_update' : SDSA,
      'sdsa_svc' : SDSA,
      'sdsa_svc_not_update' : SDSA,
      'sdsa_lr' : SDSA,
      'sdsa_lr_not_update' : SDSA,
      'sdsa_knn' : SDSA,
      'sdsa_knn_not_update' : SDSA,
      'sdsa_xgb': SDSA,
      'sdsa_xgb_not_update': SDSA,
      'sdsa_lgbm': SDSA,
      'sdsa_lgbm_not_update': SDSA,
      'sdsa_cat': SDSA,
      'sdsa_cat_not_update': SDSA,
}

# parameters_synthetic = {
#     'sdsa_rf': {'dataset1': {"gaussians": np.array([
#                 [99, 9, 99, 169, 200, 0],
#                 [108, 9, 99, 169, 200, 1]
#             ])},
#             'dataset2':   {"gaussians": np.array([
#                 [99, 9, 99, 169, 200, 0],
#                 [104, 16, 138, 16, 150, 0],
#                 [104, 16, 60, 16, 150, 1],
#                 [108, 9, 99, 169, 200, 1]
#             ])},
#             'dataset3': {"gaussians": np.array([
#                         [99, 9, 99, 169, 44, 25, 200, 0],
#                         [104, 16, 138, 16, 44, 25, 150, 0],
#                         [104, 16, 60, 16, 41, 25, 150, 1],
#                         [108, 9, 99, 169, 41, 25, 200, 1]
#             ])},
#             'dataset4': { "gaussians": np.array([
#                         [99, 9, 99, 169, 200, 0],
#                         [104, 16, 118, 16, 150, 0],
#                         [104, 16, 80, 16, 150, 1],
#                         [100, 9, 99, 169, 200, 1]
#                     ])}},

#         'sdsa_rf_not_update': {'dataset1': {"gaussians": np.array([
#                 [99, 9, 99, 169, 200, 0],
#                 [108, 9, 99, 169, 200, 1]
#             ])},
#             'dataset2':   {"gaussians": np.array([
#                 [99, 9, 99, 169, 200, 0],
#                 [104, 16, 138, 16, 150, 0],
#                 [104, 16, 60, 16, 150, 1],
#                 [108, 9, 99, 169, 200, 1]
#             ])},
#             'dataset3': {"gaussians": np.array([
#                         [99, 9, 99, 169, 44, 25, 200, 0],
#                         [104, 16, 138, 16, 44, 25, 150, 0],
#                         [104, 16, 60, 16, 41, 25, 150, 1],
#                         [108, 9, 99, 169, 41, 25, 200, 1]
#             ])},
#             'dataset4': { "gaussians": np.array([
#                         [99, 9, 99, 169, 200, 0],
#                         [104, 16, 118, 16, 150, 0],
#                         [104, 16, 80, 16, 150, 1],
#                         [100, 9, 99, 169, 200, 1]
#                     ])}}}

parameters = {
    'ivabc':{
        'dataset1': {
            'n_prototypes': [20, 20],
            'k': 3,
            'alpha': 0.2
        },
        'dataset2': {
            'n_prototypes': [35, 35],
            'k': 10,
            'alpha': 0.3
        },
        'dataset3': {
            'n_prototypes': [35, 35],
            'k': 10,
            'alpha': 0.3
        },
        'dataset4': {
            'n_prototypes': [35, 35],
            'k': 10,
            'alpha': 0.3
        },
    },
    'knn': {
            'dataset1': {},
            'dataset2': {},
            'dataset3': {},
            'dataset4': {}
        },
    'sdsa': {
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': SVC,  'parameters' : {}},
        'dataset2': {'k': [35, 35],'update': True, 'classifier': SVC,  'parameters' : {}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': SVC,  'parameters' : {}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': SVC,  'parameters' : {}}
    },
    'sdsa_not_update': {
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset2': {'k': [35, 35],'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': SVC,  'parameters' : {}}
    },
    'sdsa_rf': {
        'dataset1': {'k': [20, 20], 'update': True,'classifier': RandomForestClassifier, 'parameters' : {}},
        'dataset2': {'k': [35, 35], 'update': True,'classifier': RandomForestClassifier, 'parameters' : {}},
        'dataset3': {'k': [35, 35], 'update': True,'classifier': RandomForestClassifier, 'parameters' : {}},
        'dataset4': {'k': [35, 35], 'update': True,'classifier': RandomForestClassifier,'parameters' : {}}
    },
    'sdsa_rf_not_update': {
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': RandomForestClassifier,  'parameters' : {}},
        'dataset2': {'k': [35, 35],'update': False, 'classifier': RandomForestClassifier,  'parameters' : {}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': RandomForestClassifier,  'parameters' : {}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': RandomForestClassifier,  'parameters' : {}}
    },
    'sdsa_svc': {
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': SVC,  'parameters' : {}},
        'dataset2': {'k': [35, 35],'update': True, 'classifier': SVC,  'parameters' : {}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': SVC,  'parameters' : {}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': SVC,  'parameters' : {}}
    },
    'sdsa_svc_not_update': {
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset2': {'k': [35, 35],'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': SVC,  'parameters' : {}}
    },
    'sdsa_lr': {
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset2': {'k': [35, 35],'update': True, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}}
    },
    'sdsa_lr_not_update': {
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset2': {'k': [35, 35],'update': False, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}}
    },
    'sdsa_knn': {
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset2': {'k': [35, 35],'update': True, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}}
    },
    'sdsa_knn_not_update': {
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset2': {'k': [35, 35],'update': False, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}}
    },
    'sdsa_xgb': {
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset2': {'k': [35, 35], 'update': True, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': XGBClassifier, 'parameters': {}}
    },
    'sdsa_xgb_not_update': {
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset2': {'k': [35, 35], 'update': False, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': XGBClassifier, 'parameters': {}}
    },

    # 'sdsa_lgbm': {
    #     'dataset1': {'k': [20, 20], 'update': True, 'classifier': LGBMClassifier, 'parameters': {}},
    #     'dataset2': {'k': [35, 35], 'update': True, 'classifier': LGBMClassifier, 'parameters': {}},
    #     'dataset3': {'k': [35, 35], 'update': True, 'classifier': LGBMClassifier, 'parameters': {}},
    #     'dataset4': {'k': [35, 35], 'update': True, 'classifier': LGBMClassifier, 'parameters': {}}
    # },
    # 'sdsa_lgbm_not_update': {
    #     'dataset1': {'k': [20, 20], 'update': False, 'classifier': LGBMClassifier, 'parameters': {}},
    #     'dataset2': {'k': [35, 35], 'update': False, 'classifier': LGBMClassifier, 'parameters': {}},
    #     'dataset3': {'k': [35, 35], 'update': False, 'classifier': LGBMClassifier, 'parameters': {}},
    #     'dataset4': {'k': [35, 35], 'update': False, 'classifier': LGBMClassifier, 'parameters': {}}
    # },
    #
    # 'sdsa_cat': {
    #     'dataset1': {'k': [20, 20], 'update': True, 'classifier': CatBoostClassifier, 'parameters': {}},
    #     'dataset2': {'k': [35, 35], 'update': True, 'classifier': CatBoostClassifier, 'parameters': {}},
    #     'dataset3': {'k': [35, 35], 'update': True, 'classifier': CatBoostClassifier, 'parameters': {}},
    #     'dataset4': {'k': [35, 35], 'update': True, 'classifier': CatBoostClassifier, 'parameters': {}}
    # },
    # 'sdsa_cat_not_update': {
    #     'dataset1': {'k': [20, 20], 'update': False, 'classifier': CatBoostClassifier, 'parameters': {}},
    #     'dataset2': {'k': [35, 35], 'update': False, 'classifier': CatBoostClassifier, 'parameters': {}},
    #     'dataset3': {'k': [35, 35], 'update': False, 'classifier': CatBoostClassifier, 'parameters': {}},
    #     'dataset4': {'k': [35, 35], 'update': False, 'classifier': CatBoostClassifier, 'parameters': {}}
    # }

}


# mc_iterations = 1
# n_folds = 10


# if __name__ == '__main__':
#     np.random.seed(1)
#     #for dataset_name in parameters['sdsa_rf']:
#     dataset_name = 'dataset1'
#     params = parameters['sdsa_rf'][dataset_name]
#     dataset =  generate_multivariate_gaussians(params["gaussians"], 10)
#     # df_pd = pd.DataFrame(dataset.data)
#     # df_pd['target'] = dataset.target
#     # print(dataset.data, dataset.target)
#     X, y = dataset.data, dataset.target
#     skf = StratifiedShuffleSplit(test_size=0.5)
#     for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#         if i == 0:
#             X_train, y_train = X[train_index], y[train_index]
#             X_test, y_test = X[test_index], y[test_index]
#             sdsa = SDSA(params["k"], params["update"], params["classifier"],params['parameters'])
#             sdsa.fit(X_train, y_train)

#         # if dataset_name == "dataset3":
#         #     df = pd.DataFrame(data=X,columns=["x1_min","x1_max","x2_min","x2_max","x3_min","x3_max"])
#         #     df['target'] = y
#         #     df.to_csv("synthetic_datasets\synthetic-{}.csv".format(dataset_name), index=False)
#         # else:
#         #     df = pd.DataFrame(data=X,columns=["x1_min","x1_max","x2_min","x2_max"])
#         #     df['target'] = y
#         #     # print(df)
#         #     df.to_csv("synthetic_datasets\synthetic-{}.csv".format(dataset_name), index=False)

columns = ['dataset', 'n_classes', 'n_features', 'n_samples', 'classifier', 'mc',
           'test_fold', 'acc', 'exec_time']


def comma_separated_strings(s):
    try:
        return s.split(',')
    except ValueError:
        msg = "Not a valid comma separated list: {}".format(s)
        raise argparse.ArgumentTypeError(msg)


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs all the experiments
                                     with the given arguments''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--classifiers', dest='classifier_names',
                        type=comma_separated_strings,
                        default=['sdsa', 'sdsa_not_update', 'sdsa_rf', 'sdsa_rf_not_update',
                         'sdsa_svc','sdsa_svc_not_update','sdsa_lr', 'sdsa_lr_not_update', 'sdsa_xgb', 'sdsa_xgb_not_update',
                         'sdsa_knn_not_update', 'sdsa_knn'],
                        help='''Classifiers to use for evaluation in a comma
                        separated list of strings. From the following
                        options: ''' + ', '.join(classifiers.keys()))
    parser.add_argument('-i', '--iterations', dest='mc_iterations', type=int,
                        default=10,
                        help='Number of Monte Carlo iterations')
    parser.add_argument('-f', '--folds', dest='n_folds', type=int,
                        default=10,
                        help='Folds to create for cross-validation')
    parser.add_argument('-o', '--output-path', dest='results_path', type=str,
                        default='results_test',
                        help='''Path to store all the results''')
    parser.add_argument('-d', '--datasets', dest='datasets',
                        type=comma_separated_strings,
                        default=['dataset1', 'dataset2', 'dataset3', 'dataset4'],
                        help='''Comma separated dataset names''')
    parser.add_argument('-w', '--workers', dest='n_workers', type=int,
                        default=-1,
                        help='''Number of jobs to run concurrently. -1 to use all
                                available CPUs''')
    parser.add_argument('--distance', dest='distance', type=str,
                        default=['euclidean', 'sqeuclidean','cityblock', 'hausdorff'],
                        help='''Distance metric to use. Options are: Euclidean, City_Block, Hausdorff. Default is Euclidean.''')
    return parser.parse_args()


def compute_all(args):
    ''' Train a classifier with the specified dataset

    Parameters
    ----------
    args is a tuple with all the following:

    dataset : string
        Name of the dataset to use
    n_folds : int
        Number of folds to perform n-fold-cross-validation to train and test
        the classifier.
    mc : int
        Monte Carlo repetition index, in order to set different seeds to
        different repetitions, but same seed in calibrators in the same Monte
        Carlo repetition.
    classifier_name : string
        Name of the classifier to be trained and tested

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the overall results of every calibration method
        dataset : string
            Name of the dataset
        classifier : string
            Classifier name
        n_classes : int
            Number of classes in the dataset
        n_features : int
            Number of features in the dataset
        n_samples : int
            Number of samples in the dataset
        mc : int
            Monte Carlo repetition index
        acc : float
            Mean accuracy
        exec_time : float
            Mean execution time
    '''
    (dataset, n_folds, mc, classifier_name, results_path, distance) = args

    classifier = classifiers[classifier_name]
    params = parameters[classifier_name][dataset]


    # data =  generate_multivariate_gaussians(parameters_synthetic[classifier_name][dataset]["gaussians"], 10)
    # X, y = data.data, data.target

    # if dataset == 'dataset3':
    #     df = pd.DataFrame(data=X,columns=["x1_min","x1_max","x2_min","x2_max","x3_min","x3_max"])
    #     df['target'] = y
    # else:
    #     df = pd.DataFrame(data=X,columns=["x1_min","x1_max","x2_min","x2_max"])
    #     df['target'] = y
    #     df.to_csv("synthetic_datasets\synthetic-{}.csv".format(dataset), index=False)
    data = pd.read_csv('./synthetic_datasets/synthetic-{}.csv'.format(dataset)) 
    # X, y = data.data, data.target
    # params = parameters['sdsa_rf'][dataset]
    params = parameters[classifier_name][dataset]
    params['dist'] = distance

    # data =  generat e_multivariate_gaussians(parameters_synthetic[classifier_name][dataset]["gaussians"], 10)

    # X, y = data.data, data.target


    X = data.drop('target', axis=1).values                                                                                                                                         

    y = data['target'].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=mc)
    n_classes = len(np.unique(y))
    
    results = []

    fold_id = 0
    for train_idx, test_idx in skf.split(X, y):
        print(
            'Computing: classifier: {}, dataset: {}, dist: {}, mc: {} fold: {}'.format(
                classifier_name, dataset, distance, mc, fold_id
            )
        )
        x_train, y_train = X[train_idx], y[train_idx]
        x_test, y_test = X[test_idx], y[test_idx]

        c = classifier(**params)
        # c = classifier(params["k"], params["update"], params["classifier"],params['parameters'])
        start = time.time()
        c.fit(x_train, y_train)
        end = time.time()

        exec_time = end - start

        acc = c.accuracy(x_test, y_test)
        results.append(
            [dataset, n_classes, X.shape[1]/2, X.shape[0], classifier_name, mc,
           fold_id, acc, exec_time]
        )

        fold_id += 1
    df = pd.DataFrame(data=results, columns=columns)
    df.to_csv(os.path.join(results_path, 'dataset-{}-dist-{}-mc-{}.csv'.format(dataset, distance, mc)))
    return df


def main(mc_iterations, n_folds, classifier_names, results_path,
		 datasets, n_workers):

    dataset_names = datasets
    dataset_names.sort()

    classifier_names.sort()
    results_path_root = results_path

    for classifier_name in classifier_names:
        all_results = []
        results_path = os.path.join(results_path_root, classifier_name)

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        for dataset in dataset_names:
            mcs = np.arange(mc_iterations)

            args = [[dataset], [n_folds], mcs, [classifier_name], [results_path]]
            args = list(itertools.product(*args))

            if n_workers == -1:
                n_workers = cpu_count()

            if n_workers == 1:
                map_f = map
            else:
                if n_workers > len(args):
                    n_workers = len(args)

                p = Pool(n_workers)
                map_f = p.map

            dfs = map_f(compute_all, args)
            all_results.extend(dfs)
        all_results = pd.concat(all_results)
        all_results.to_csv(
            os.path.join(
                results_path, '{}.csv'.format(','.join(dataset_names))
            )
        )


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
