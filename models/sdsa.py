import scipy.sparse as sp
# from sklearn.datasets import load_iris
import random
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import stable_cumsum
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression


class SDSA:

    def __init__ (self, k, update = True, classifier = DecisionTreeClassifier, parameters = None):
        self.k = k
        self.update = update
        self.classifier = classifier
        self.parameters = parameters

    def fit(self, X, y):    
        #treinamento
        # sort = np.random.choice(range(len(X)), self.k)
        #print(sort)
        # n = X[sort]
        #print(n)

        # distance = cdist(X, n)
        # C  = np.argmin(distance, axis=1)
        #print(C)

        # medias = []
        # for c in range(self.k): 
        #     if np.any(C==c):
        #         medias.append(np.mean(X[(C == c)], axis = 0))
        # medias = np.array(medias)

        # medias_min = medias[:,::2]
        # medias_max = medias[:,1::2]
        #print(medias)

        media_prot = []
        
        for i, n_prots in enumerate(self.k):  
            X_class = X[y==i]
            medias = kmeanspp(X_class, n_prots)

            if self.update == True:
                for t in range(100):

                    # d_min = cdist(X[:,::2], medias_min)
                    # d_max = cdist(X[:,1::2], medias_max)    
                    # distance = d_min + d_max
                    D = distances(X_class, medias)
                    C  = np.argmin(D, axis=1)

                    #print(distance)
                    #print(C)
                    # minimos = np.min(distance, axis=1)
                    #print(minimos)

                    # medias = []
                    # for c in range(self.k): 
                    #     if np.any(C==c):
                    #       medias.append(np.mean(X[(C == c)], axis = 0))                    
                    # medias = np.array(medias)

                    for c in range(n_prots): 
                        if np.any(C==c):
                            medias[c] = np.mean(X_class[C == c], axis = 0)     
                    # medias_min = medias[:,::2]
                    # medias_max = medias[:,1::2]
                    # d_min = cdist(X[:,::2], medias_min)
                    # d_max = cdist(X[:,1::2], medias_max)
                    # D = d_min + d_max
            
            # d_min = cdist(X[:,::2], medias_min)
            # d_max = cdist(X[:,1::2], medias_max)
            # D = d_min + d_max        
            media_prot.append(medias)  

        prots = np.vstack(media_prot)

        D = distances(X, prots)
   
        clf = self.classifier(**self.parameters)

        clf.fit(D, y)
        self.clf = clf
        self.medias = prots
        return self

    def accuracy(self, X, Y):
        #teste e retorna acuracia
        #predições para o teste

        # D_min = cdist(X[:, ::2], self.medias[:,::2])
        # D_max = cdist(X[:, 1::2], self.medias[:,1::2])

        # D = D_min + D_max

        D = distances(X, self.medias)
        predicoes = self.clf.predict(D)
        
        accuracy = np.sum(predicoes == Y)/len(predicoes == Y)
        return accuracy    


def distances(matrix1, matrix2):
    d_min = cdist(matrix1[:,::2], matrix2[:,::2])
    d_max = cdist(matrix1[:,1::2], matrix2[:,1::2])
    return d_min + d_max


def kmeanspp(X, n_clusters, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : integer
        The number of seeds to choose

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = np.random.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = distances(centers[0, np.newaxis], X)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = distances(X[candidate_ids], X)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers
