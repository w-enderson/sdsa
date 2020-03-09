import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from utils import select_prototypes

class WFLVQ:
    def __init__(self, n_prototypes=None, learning_rate=0.3, n_iter=500,
                 n_iter_without_progress=3):
        self.n_prototypes_ = n_prototypes
        self.learning_rate_ = learning_rate
        self.n_iter_ = n_iter
        self.n_iter_without_progress_ = n_iter_without_progress

    def fit(self, X, y):
        eps = 1e-100

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        train_index, valid_index = next(sss.split(X, y))

        train_X, train_y = X[train_index], y[train_index]
        valid_X, valid_y = X[valid_index], y[valid_index]

        prototypes_X, prototypes_y = select_prototypes(
            train_X,
            train_y,
            self.n_prototypes_
        )

        k, n, p = len(prototypes_y), len(train_X), int(prototypes_X.shape[1] / 2)

        t, t_without_progress = 0, 0

        feature_weights = np.ones((k, p))
        prototype_weights = np.ones(k)
        deltas = np.zeros((k, p))

        saved_prototypes_X = np.copy(prototypes_X)
        saved_prototypes_y = np.copy(prototypes_y)
        saved_weights = np.copy(feature_weights)

        validation_error = np.inf

        while t_without_progress < self.n_iter_without_progress_ and t < self.n_iter_:
            cycles = 0
            rates = np.ones((k, 1)) * self.learning_rate_
            while cycles < k * n:
                i = np.random.choice(n)
                x = train_X[i]        

                dists = calculate_distances_(
                    x.reshape(1, -1), prototypes_X, feature_weights
                )

                deg = calculate_degrees_(dists).flatten()

                correct = train_y[i] == prototypes_y

                # feature weights update step
                mins = (prototypes_X[correct,::2] - x[::2])**2
                maxs = (prototypes_X[correct,1::2] - x[1::2])**2
                temp = deg[correct].reshape(-1, 1) * (mins + maxs)
                deltas[correct] = np.clip(
                    (1.0 - rates[correct]) * deltas[correct] + rates[correct] * temp,
                    eps,
                    np.inf
                )

                prod = (
                    prototype_weights[correct] * np.prod(deltas[correct], axis=1)
                ).reshape(-1, 1) ** (1 / p)
                feature_weights[correct] = prod / deltas[correct]

                # features update step
                correct = (2 * correct - 1.0).reshape(-1, 1)
                deg = deg.reshape(-1, 1)
                prototypes_X += correct * deg * rates * (x - prototypes_X)
                rates = rates / (1.0 + (correct*rates))

                rates = np.clip(rates, 0.00005, 0.999999)
                if np.sum(rates == 0.00005) == k:
                    break
                cycles += 1
            t += 1 
            new_error = 1. - accuracy_(
                valid_X, valid_y, prototypes_X, prototypes_y, feature_weights
            )
            if new_error < validation_error:
                t_without_progress = 0
                saved_prototypes_X = np.copy(prototypes_X)
                saved_prototypes_y = np.copy(prototypes_y)
                saved_weights = np.copy(feature_weights)
            else:
                t_without_progress += 1
            prototype_weights = calculate_prototype_weights_(
                X, y, prototypes_X, prototypes_y, feature_weights
            )
            validation_error = new_error

        self.prototypes_ = (
            saved_prototypes_X,
            saved_prototypes_y
        )
        self.feature_weights_ = saved_weights
        return self

    def predict(self, X):

        return predict_(
            X, 
            self.prototypes_[0], 
            self.prototypes_[1], 
            self.feature_weights_
        )

    def accuracy(self, X, y):
        
        return accuracy_(
            X, 
            y, 
            self.prototypes_[0], 
            self.prototypes_[1], 
            self.feature_weights_
        )

def calculate_distances_(X, prototypes_X, weights):
    w = weights

    mins_prots, mins_X = prototypes_X[:,::2], X[:,::2]
    maxs_prots, maxs_X = prototypes_X[:,1::2], X[:,1::2]

    d_mins = ((
        (mins_prots - mins_X[:, np.newaxis]) ** 2
    ) * w).sum(axis=2)
    d_maxs = ((
        (maxs_prots - maxs_X[:, np.newaxis]) ** 2
    ) * w).sum(axis=2)

    return d_mins + d_maxs


def calculate_degrees_(distances, eps=1e-10):
    d = (distances + eps) ** -2
    return d / d.sum(axis=1).reshape(-1, 1)


def calculate_criterion_(distances, degrees, prototypes_y, y):
    partition = np.argmax(degrees, axis=1)  
    predictions = prototypes_y[partition] 
    wrong = predictions != y

    degrees[wrong,:] = 0
    return np.sum(degrees[:, partition] * distances[:, partition])

def predict_(X, prototypes_X, prototypes_y, weights):
    distances = calculate_distances_(X, prototypes_X, weights)
    degrees = calculate_degrees_(distances)
    partition = np.argmax(degrees, axis=1) 
    predictions = prototypes_y[partition] 
    
    return predictions


def accuracy_(X, y, prototypes_X, prototypes_y, weights):
    predictions = predict_(X, prototypes_X, prototypes_y, weights)

    return np.mean(predictions == y)


def calculate_prototype_weights_(X, y, prototypes_X, prototypes_y, weights):
    n_labels = len(np.unique(prototypes_y))

    distances = calculate_distances_(X, prototypes_X, weights)
    degrees = calculate_degrees_(distances)
    partition = np.argmax(degrees, axis=1) 
    predictions = prototypes_y[partition] 

    wrong = predictions != y
    degrees[wrong] = 0

    ind = range(len(partition))
    deg_dis = (degrees[ind,partition] * distances[ind,partition]).reshape(-1,1)

    pred = np.repeat(
        predictions.reshape(-1, 1), n_labels, axis=1
    ) == np.arange(n_labels)

    class_totals = (
        np.repeat(deg_dis, n_labels, axis=1) * pred
    ).sum(axis=0) + 1
    prod_classes = np.prod(class_totals) ** (1 / n_labels)
    class_weights = prod_classes / class_totals

    k = len(prototypes_y)
    pred = np.repeat(partition.reshape(-1, 1), k, axis=1) == np.arange(k)
    prot_totals = (
        np.repeat(deg_dis, k, axis=1) * pred
    ).sum(axis=0) + 1

    prt = np.repeat(prot_totals.reshape(1, -1), k, axis=0)
    same_class = prototypes_y.reshape(1, -1) == prototypes_y.reshape(-1, 1)
    temp = prt * same_class
    temp[temp == 0] = 1

    prod_prots = (
        class_weights[prototypes_y] * temp.prod(axis=1)
     ) ** (1 / np.bincount(prototypes_y)[prototypes_y])
    
    return prod_prots / prot_totals
