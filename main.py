import argparse
import time
import os

# Parallelization
import itertools
from multiprocessing import cpu_count, Pool

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from models.wflvq import WFLVQ
from models.ivabc import IVABC

classifiers = {
      'wflvq': WFLVQ,
      'ivabc': IVABC
}

parameters = {
    'wflvq': {
        'climates': {'n_prototypes': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40]},
        'dry-climates': {'n_prototypes': [28, 38, 34]},
        'european-climates': {'n_prototypes': [20, 40]},
        'mushroom': {'n_prototypes': [7, 2]}
    },
    'ivabc': {
        'climates': {
            'n_prototypes': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40],
            'k': 8,
            'alpha': 1.0
        },
        'dry-climates': {
            'n_prototypes': [28, 38, 34],
            'k': 28,
            'alpha': 0.5
        },
        'european-climates': {
            'n_prototypes': [20, 40],
            'k': 3,
            'alpha': 0.2
        },
        'mushroom': {
            'n_prototypes': [7, 2],
            'k': 3,
            'alpha': 0.0
        }
    }
}

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
                        default=['wflvq', 'ivabc'],
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
                        default=['climates', 'dry-climates', 'european-climates', 'mushroom'],
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
    (dataset, n_folds, mc, classifier_name, results_path) = args

    classifier = classifiers[classifier_name]
    params = parameters[classifier_name][dataset]
    
    data = pd.read_csv('./datasets/{}.csv'.format(dataset)) 

    X = data.drop('target', axis=1).values                                                                                                                                         

    y = data['target'].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=mc)
    n_classes = len(np.unique(y))
    
    results = []

    fold_id = 0
    for train_idx, test_idx in skf.split(X, y):
        print(
            'Computing: classifier: {}, dataset: {}, mc: {} fold: {}'.format(
                classifier_name, dataset, mc, fold_id
            )
        )
        x_train, y_train = X[train_idx], y[train_idx]
        x_test, y_test = X[test_idx], y[test_idx]

        c = classifier(**params)
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
    df.to_csv(os.path.join(results_path, 'dataset-{}-mc-{}.csv'.format(dataset, mc)))
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