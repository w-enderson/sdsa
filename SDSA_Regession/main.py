import argparse
import time
import os

# Parallelization
import itertools
from multiprocessing import cpu_count, Pool

import pandas as pd
import numpy as np


from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

from models.wflvq import WFLVQ
from models.ivabc import IVABC
from models.sdsr import SDSR
from models.sdsr import CenterRangeSVR
from models.sdsr import CenterRangeRF
from models.sdsr import CenterRangeLinear
from models.sdsr import CenterRangeOLS
from models.sdsr import CenterRangeLinearComparition

regressors = {
      'sdsr_svr' : SDSR,
      'sdsr_svr_not_update' : SDSR,
      'sdsr_rf' : SDSR,
      'sdsr_rf_not_update': SDSR,
      'sdsr_linear': SDSR,
      'sdsr_linear_not_update': SDSR,
      'sdsr_ols': SDSR,
      'sdsr_ols_not_update': SDSR,
      'sdsr_linear_comparition': SDSR,
      'sdsr_linear_comparition_not_update': SDSR
}

parameters = {
    'sdsr_svr': {
        'climates': {'k': 10, 'regressor': CenterRangeSVR, 'parameters' : {}},    
        'akc-data': {'k': 7, 'regressor': CenterRangeSVR, 'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeSVR, 'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeSVR, 'parameters' : {}}
    },
    'sdsr_svr_not_update': {
        'climates': {'k': 10, 'regressor': CenterRangeSVR, 'update': False, 'parameters' : {}},
        'akc-data': {'k': 7, 'regressor': CenterRangeSVR, 'update': False, 'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeSVR, 'update': False, 'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeSVR, 'update': False, 'parameters' : {}}
    },
    'sdsr_rf': {
        'climates': {'k': 10, 'regressor': CenterRangeRF, 'parameters' : {}},
        'akc-data': {'k': 7, 'regressor': CenterRangeRF,  'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeRF, 'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeRF,  'parameters' : {}}
    },
    'sdsr_rf_not_update': {
        'climates': {'k': 10, 'regressor': CenterRangeRF, 'update': False, 'parameters' : {}},
        'akc-data': {'k': 7, 'regressor': CenterRangeRF, 'update': False, 'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeRF, 'update': False, 'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeRF,  'update': False, 'parameters' : {}}
    },
    'sdsr_linear': {
        'climates': {'k': 10, 'regressor': CenterRangeLinear, 'parameters' : {}},
        'akc-data': {'k': 10, 'regressor': CenterRangeLinear,  'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeLinear, 'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeLinear,  'parameters' : {}}
    },
    'sdsr_linear_not_update': {
        'climates': {'k': 10, 'regressor': CenterRangeLinear, 'update': False, 'parameters' : {}},
        'akc-data': {'k': 7, 'regressor': CenterRangeLinear, 'update': False,  'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeLinear, 'update': False,  'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeLinear,  'update': False, 'parameters' : {}}
    },
       'sdsr_ols': {
        'climates': {'k': 10, 'regressor': CenterRangeOLS, 'parameters' : {}},
        'akc-data': {'k': 7, 'regressor': CenterRangeOLS,  'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeOLS, 'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeOLS,  'parameters' : {}}
    },
    'sdsr_ols_not_update': {
        'climates': {'k': 10, 'regressor': CenterRangeOLS, 'update': False, 'parameters' : {}},
        'akc-data': {'k': 10, 'regressor': CenterRangeOLS,  'update': False, 'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeOLS, 'update': False, 'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeOLS, 'update': False, 'parameters' : {}}
    },
     'sdsr_linear_comparition': {
        'climates': {'k': 10, 'regressor': CenterRangeLinearComparition, 'parameters' : {}},
        'akc-data': {'k': 7, 'regressor': CenterRangeLinearComparition,  'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeLinearComparition, 'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeLinearComparition,  'parameters' : {}}
    },
     'sdsr_linear_comparition_not_update': {
        'climates': {'k': 10, 'regressor': CenterRangeLinearComparition, 'update': False, 'parameters' : {}},
        'akc-data': {'k': 7, 'regressor': CenterRangeLinearComparition,  'update': False, 'parameters' : {}},
        'scientific-production': {'k': 10, 'regressor': CenterRangeLinearComparition, 'update': False, 'parameters' : {}},
        'mushroom': {'k': 2,'regressor': CenterRangeLinearComparition,  'update': False, 'parameters' : {}}
    }
}

columns = ['dataset', 'n_classes', 'n_features', 'n_samples', 'regression_model', 'mc',
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
    parser.add_argument('-c', '--regressors', dest='regressors_names',
                        type=comma_separated_strings,
                        default=['wflvq', 'ivabc','sdsr_svr', 'sdsr_svr_not_update','sdsr_rf', 'sdsr_rf_not_update',
                        'sdsr_linear','sdsr_linear_not_update','sdsr_ols','sdsr_ols_not_update','sdsr_linear_comparition',
                        'sdsr_linear_comparition_not_update'],
                        help='''Classifiers to use for evaluation in a comma
                        separated list of strings. From the following
                        options: ''' + ', '.join(regressors.keys()))
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
                        default=['climates', 'akc-data', 'scientific-production', 'mushroom'],
                        help='''Comma separated dataset names''')
    parser.add_argument('-w', '--workers', dest='n_workers', type=int,
                        default=-1,
                        help='''Number of jobs to run concurrently. -1 to use all
                                available CPUs''')
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
    (dataset, n_folds, mc, regression_model_name, results_path) = args

    regressor = regressors[regression_model_name]
    params = parameters[regression_model_name][dataset]
    
    data = pd.read_csv('./datasets/{}.csv'.format(dataset)) 

    X = data.drop(['y_min','y_max'], axis=1).values                                                                                                                                         

    y = data[['y_min','y_max']].values

    skf = KFold(n_splits=n_folds, shuffle=True, random_state=mc)
    n_classes = len(np.unique(y))
    
    results = []

    fold_id = 0
    for train_idx, test_idx in skf.split(X):
        print(
            'Computing: regressor: {}, dataset: {}, mc: {} fold: {}'.format(
                regression_model_name, dataset, mc, fold_id
            )
        )
        x_train, y_train = X[train_idx], y[train_idx]
        x_test, y_test = X[test_idx], y[test_idx]

        c = regressor(**params)
        start = time.time()
        c.fit(x_train, y_train)
        end = time.time()

        exec_time = end - start

        #acc = c.accuracy(x_test, y_test)
        #rsquare = c.r_square(x_test, y_test)
        mmre = c.mmre(x_test, y_test)
        results.append(
            [dataset, n_classes, X.shape[1]/2, X.shape[0], regression_model_name, mc,
           fold_id, mmre, exec_time]
        )

        fold_id += 1
    df = pd.DataFrame(data=results, columns=columns)
    df.to_csv(os.path.join(results_path, 'dataset-{}-mc-{}.csv'.format(dataset, mc)))
    return df


def main(mc_iterations, n_folds, regressors_names, results_path,
		 datasets, n_workers):

    dataset_names = datasets
    dataset_names.sort()

    regressors_names.sort()
    results_path_root = results_path

    for regression_model_name in regressors_names:
        all_results = []
        results_path = os.path.join(results_path_root, regression_model_name)

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        for dataset in dataset_names:
            mcs = np.arange(mc_iterations)

            args = [[dataset], [n_folds], mcs, [regression_model_name], [results_path]]
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
                #map_f = map

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