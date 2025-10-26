import scipy.sparse as sp
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import stable_cumsum

from sklearn.preprocessing import StandardScaler


class SDSA:

    def __init__ (self, k, dist='euclidean',update = True, classifier = DecisionTreeClassifier, parameters = None):
        self.k = k
        self.update = update
        self.classifier = classifier
        self.parameters = parameters
        self.dist = dist
    
    def get_prototypes(self, X, y):
        media_prot = []
        
        for i, n_prots in enumerate(self.k):  
            X_class = X[y==i]
            medias = kmeanspp(X_class, n_prots)

            if self.update:
                for t in range(100):

                    D = distance(X_class, medias, dist=self.dist)
                    C  = np.argmin(D, axis=1)

                    for c in range(n_prots): 
                        if np.any(C==c):
                            medias[c] = np.mean(X_class[C == c], axis = 0)     
             
            media_prot.append(medias)  

        prots = np.vstack(media_prot)

        return prots
        

    def fit(self, X, y):    

        prots = self.get_prototypes(X, y)

        D = distance(X, prots, dist=self.dist)
        
        self.scaler = StandardScaler()
        D_scaled = self.scaler.fit_transform(D)

        clf = self.classifier(**self.parameters)
   
        clf.fit(D_scaled, y)

        self.clf = clf
        self.medias = prots
        return self


    
    def accuracy(self, X, Y):

        D = distance(X, self.medias, self.dist)
        D_scaled = self.scaler.transform(D)

        predicoes = self.clf.predict(D_scaled)

        accuracy = np.sum(predicoes == Y)/len(predicoes == Y)
        return accuracy




def distance(matrix1, matrix2, dist):
    if dist not in ['euclidean', 'sqeuclidean', 'cityblock', 'hausdorff']:
        raise ValueError(f"Distância não permitida: {dist}")
    
    distancia = dist if dist!="hausdorff" else "cityblock"

    d_min = cdist(matrix1[:,::2], matrix2[:,::2], metric= distancia)
    d_max = cdist(matrix1[:,1::2], matrix2[:,1::2], metric= distancia)

    d_matrix = np.maximum(d_min, d_max) if dist=="hausdorff" else d_min + d_max

    return d_matrix



# Interval k-NN classifier used to live here; implementation was moved to models/knn.py
# to avoid duplication. Use `from models.knn import IntervalKNN` when you need the class.


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
    closest_dist_sq = distance(centers[0, np.newaxis], X, dist='euclidean')
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = distance(X[candidate_ids], X, dist='euclidean')

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
