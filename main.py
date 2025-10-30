import argparse
import time
import os

import itertools
from multiprocessing import cpu_count, Pool

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


from models.wflvq import WFLVQ
from models.ivabc import IVABC
from models.sdsa import SDSA
from models.knn import IntervalKNN


classifiers = {
      'wflvq': WFLVQ,
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
}

parameters = {
    'wflvq': {
        'climates': {'n_prototypes': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40]},
        'dry_climates': {'n_prototypes': [28, 38, 34]},
        'european_climates': {'n_prototypes': [20, 40]},
        'mushroom': {'n_prototypes': [7, 2]}
    },
    'ivabc': {
        'climates': {
            'n_prototypes': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40],
            'k': 8,
            'alpha': 1.0
        },
        'dry_climates': {
            'n_prototypes': [28, 38, 34],
            'k': 28,
            'alpha': 0.5
        },
        'european_climates': {
            'n_prototypes': [20, 40],
            'k': 3,
            'alpha': 0.2
        },
        'mushroom': {
            'n_prototypes': [7, 2],
            'k': 3,
            'alpha': 0.0
        },
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
        'dataset5': {
            'n_prototypes': [20, 20],
            'k':3,
            'alpha':0.2

        }
    },
    'knn': {
        'climates': {},
        'dry_climates': {},
        'european_climates': {},
        'mushroom': {},
        'dataset1': {},
        'dataset2': {},
        'dataset3': {},
        'dataset4': {},
        'dataset5': {}
    },
    'sdsa': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'parameters' : {}},
        'dry_climates': {'k': [28, 38, 34], 'parameters' : {}},
        'european_climates': {'k': [20, 40], 'parameters' : {}},
        'mushroom': {'k': [7, 2], 'parameters' : {}},
        'dataset1': {'k': [20, 20], 'update': True, 'parameters' : {}},
        'dataset2': {'k': [35, 35],'update': True, 'parameters' : {}},
        'dataset3': {'k': [35, 35], 'update': True, 'parameters' : {}},
        'dataset4': {'k': [35, 35], 'update': True, 'parameters' : {}},
        'dataset5': {'k': [20, 20], 'update': True, 'parameters' : {}},

    },
    'sdsa_not_update': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'update': False, 'parameters' : {}},
        'dry_climates': {'k': [28, 38, 34], 'update': False, 'parameters' : {}},
        'european_climates': {'k': [20, 40], 'update': False, 'parameters' : {}},
        'mushroom': {'k': [7, 2],'update': False, 'parameters' : {}},
        'dataset1': {'k': [20, 20], 'update': False, 'parameters' : {}},
        'dataset2': {'k': [35, 35],'update': False, 'parameters' : {}},
        'dataset3': {'k': [35, 35], 'update': False, 'parameters' : {}},
        'dataset4': {'k': [35, 35], 'update': False, 'parameters' : {}},
        'dataset5': {'k': [20, 20], 'update': False, 'parameters' : {}},
    },
    'sdsa_rf': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': RandomForestClassifier, 'parameters' : {}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': RandomForestClassifier, 'parameters' : {}},
        'european_climates': {'k': [20, 40], 'classifier': RandomForestClassifier, 'parameters' : {}},
        'mushroom': {'k': [7, 2],'classifier': RandomForestClassifier, 'parameters' : {}},
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': RandomForestClassifier, 'parameters': {}},
        'dataset2': {'k': [35, 35], 'update': True, 'classifier': RandomForestClassifier, 'parameters': {}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': RandomForestClassifier, 'parameters': {}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': RandomForestClassifier, 'parameters': {}},
        'dataset5': {'k': [20, 20], 'update': True, 'classifier': RandomForestClassifier, 'parameters': {}},
    },
    'sdsa_rf_not_update': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': RandomForestClassifier, 'update': False, 'parameters' : {}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': RandomForestClassifier, 'update': False, 'parameters' : {}},
        'european_climates': {'k': [20, 40], 'classifier': RandomForestClassifier, 'update': False, 'parameters' : {}},
        'mushroom': {'k': [7, 2],'classifier': RandomForestClassifier, 'update': False, 'parameters' : {}},
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': RandomForestClassifier, 'parameters': {}},
        'dataset2': {'k': [35, 35], 'update': False, 'classifier': RandomForestClassifier, 'parameters': {}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': RandomForestClassifier, 'parameters': {}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': RandomForestClassifier, 'parameters': {}},
        'dataset5': {'k': [20, 20], 'update': False, 'classifier': RandomForestClassifier, 'parameters': {}},
    },
    'sdsa_svc': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': SVC, 'parameters' : {}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': SVC, 'parameters' : {}},
        'european_climates': {'k': [20, 40], 'classifier': SVC, 'parameters' : {}},
        'mushroom': {'k':[7, 2],'classifier': SVC, 'parameters' : {}},
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': SVC, 'parameters': {}},
        'dataset2': {'k': [35, 35], 'update': True, 'classifier': SVC, 'parameters': {}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': SVC, 'parameters': {}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': SVC, 'parameters': {}},
        'dataset5': {'k': [20, 20], 'update': True, 'classifier': SVC, 'parameters': {}},
    },
    'sdsa_svc_not_update': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': SVC, 'update': False, 'parameters' : {}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': SVC, 'update': False, 'parameters' : {}},
        'european_climates': {'k': [20, 40], 'classifier': SVC, 'update': False, 'parameters' : {}},
        'mushroom': {'k': [7, 2],'classifier': SVC, 'update': False, 'parameters' : {}},
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset2': {'k': [35, 35],'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': SVC,  'parameters' : {}},
        'dataset5': {'k': [20, 20], 'update': False, 'classifier': SVC,  'parameters' : {}},
    },
    'sdsa_lr': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': LogisticRegression, 'parameters' : {'max_iter' : 120000}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': LogisticRegression, 'parameters' : {'max_iter' : 120000}},
        'european_climates': {'k': [20, 40], 'classifier': LogisticRegression, 'parameters' : {'max_iter' : 120000}},
        'mushroom': {'k': [7, 2],'classifier': LogisticRegression, 'parameters' : {'max_iter' : 120000}},
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': LogisticRegression,'parameters': {'max_iter': 120000}},
        'dataset2': {'k': [35, 35], 'update': True, 'classifier': LogisticRegression,'parameters': {'max_iter': 120000}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': LogisticRegression,'parameters': {'max_iter': 120000}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': LogisticRegression,'parameters': {'max_iter': 120000}},
        'dataset5': {'k': [20, 20], 'update': True, 'classifier': LogisticRegression,'parameters': {'max_iter': 120000}},
    },
    'sdsa_lr_not_update': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': LogisticRegression,  'update': False, 'parameters' : {'max_iter' : 120000}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': LogisticRegression,  'update': False, 'parameters' : {'max_iter' : 120000}},
        'european_climates': {'k': [20, 40], 'classifier': LogisticRegression,  'update': False, 'parameters' : {'max_iter' : 120000}},
        'mushroom': {'k': [7, 2],'classifier': LogisticRegression,  'update': False, 'parameters' : {'max_iter' : 120000}},
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset2': {'k': [35, 35],'update': False, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
        'dataset5': {'k': [20, 20], 'update': False, 'classifier': LogisticRegression,  'parameters' : {'max_iter' : 120000}},
    },
    'sdsa_knn': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': KNeighborsClassifier, 'parameters' : {'n_neighbors' : 5}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': KNeighborsClassifier, 'parameters' : {'n_neighbors' : 5}},
        'european_climates': {'k': [20, 40], 'classifier': KNeighborsClassifier, 'parameters' : {'n_neighbors' : 5}},
        'mushroom': {'k': [7, 2], 'classifier': KNeighborsClassifier,'parameters' : {'n_neighbors' : 5}},
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset2': {'k': [35, 35],'update': True, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset5': {'k': [20, 20], 'update': True, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
    },
    'sdsa_knn_not_update': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': KNeighborsClassifier,  'update': False, 'parameters' : {'n_neighbors' : 5}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': KNeighborsClassifier,  'update': False, 'parameters' : {'n_neighbors' : 5}},
        'european_climates': {'k': [20, 40], 'classifier': KNeighborsClassifier,  'update': False, 'parameters' : {'n_neighbors' : 5}},
        'mushroom': {'k': [7, 2],'classifier': KNeighborsClassifier,  'update': False, 'parameters' : {'n_neighbors' : 5}},
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset2': {'k': [35, 35],'update': False, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
        'dataset5': {'k': [20, 20], 'update': False, 'classifier': KNeighborsClassifier,  'parameters' : {'n_neighbors' : 5}},
    },
    'sdsa_xgb': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': XGBClassifier, 'parameters': {}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': XGBClassifier, 'parameters': {}},
        'european_climates': {'k': [20, 40], 'classifier': XGBClassifier, 'parameters': {}},
        'mushroom': {'k': [7, 2], 'classifier': XGBClassifier, 'parameters': {}},
        'dataset1': {'k': [20, 20], 'update': True, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset2': {'k': [35, 35], 'update': True, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset3': {'k': [35, 35], 'update': True, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset4': {'k': [35, 35], 'update': True, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset5': {'k': [20, 20], 'update': True, 'classifier': XGBClassifier, 'parameters': {}},
    },
    'sdsa_xgb_not_update': {
        'climates': {'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': XGBClassifier, 'update': False, 'parameters': {}},
        'dry_climates': {'k': [28, 38, 34], 'classifier': XGBClassifier, 'update': False, 'parameters': {}},
        'european_climates': {'k': [20, 40], 'classifier': XGBClassifier, 'update': False, 'parameters': {}},
        'mushroom': {'k': [7, 2], 'classifier': XGBClassifier, 'update': False, 'parameters': {}},
        'dataset1': {'k': [20, 20], 'update': False, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset2': {'k': [35, 35], 'update': False, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset3': {'k': [35, 35], 'update': False, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset4': {'k': [35, 35], 'update': False, 'classifier': XGBClassifier, 'parameters': {}},
        'dataset5': {'k': [20, 20], 'update': False, 'classifier': XGBClassifier, 'parameters': {}},
    },

}
''
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
                        default=['wflvq', 'ivabc','sdsa', 'sdsa_not_update', 'sdsa_rf', 'sdsa_rf_not_update',
                         'sdsa_svc','sdsa_svc_not_update','sdsa_lr', 'sdsa_lr_not_update', 'sdsa_xgb', 'sdsa_xgb_not_update'],
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
                        default=['climates', 'dry_climates', 'european_climates', 'mushroom',
                        'dataset1', 'dataset2', 'dataset3', 'dataset4'],
                        help='''Comma separated dataset names''')
    parser.add_argument('-w', '--workers', dest='n_workers', type=int,
                        default=-1,
                        help='''Number of jobs to run concurrently. -1 to use all
                                available CPUs''')
    parser.add_argument('--distances', dest='distances', type=comma_separated_strings,
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

    if distance:
        params['dist'] = distance


    data = pd.read_csv('./datasets/{}.csv'.format(dataset)) 

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
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        c = classifier(**params)

        start = time.time()
        c.fit(X_train, y_train)
        end = time.time()

        exec_time = end - start

        acc = c.accuracy(X_test, y_test)
        results.append(
            [dataset, n_classes, X.shape[1]/2, X.shape[0], classifier_name, mc,
           fold_id, acc, exec_time]
        )

        fold_id += 1

    df = pd.DataFrame(data=results, columns=columns)

    if distance:
        filename = 'dataset-{}-dist-{}-mc-{}.csv'.format(dataset, distance, mc)
    else:
        filename = 'dataset-{}-mc-{}.csv'.format(dataset, mc)

    df.to_csv(os.path.join(results_path, filename))

    return df


def main(mc_iterations, n_folds, classifier_names, results_path,
		 datasets, n_workers, distances):

    dataset_names = datasets
    distance_names = distances

    dataset_names.sort()
    classifier_names.sort()


    results_path_root = results_path


    for classifier_name in classifier_names:
        all_results = []
        results_path = os.path.join(results_path_root, classifier_name)

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        for dataset in dataset_names:
            for d in distance_names:
                
                dist = d if "sdsa" in classifier_name else None 

                mcs = np.arange(mc_iterations)

                args = [ [dataset], [n_folds], mcs, [classifier_name], [results_path], [dist] ]
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

                if not dist:
                    break



if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))