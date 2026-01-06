import scipy.sparse as sp
import numpy as np
from scipy.spatial.distance import cdist


from sklearn.tree import DecisionTreeClassifier


from sklearn.utils.extmath import stable_cumsum
from sklearn.preprocessing import StandardScaler

from typing import Any, Type

class SDSA:

    def __init__ (self, 
                    k: list[int], 
                    dist: str ='euclidean', 
                    update: bool = True, 
                    classifier: Type[Any] = DecisionTreeClassifier, 
                    parameters: dict[str, Any] | None = None
                 ):
        '''
        k : número de protótipos por classe
        dist : distancia intervalar usada
        update :
        classifier : classificador clássico usado
        parameters : parametros do classificador
        '''

        self.k = k
        self.dist = dist
        self.update = update
        self.classifier = classifier
        self.parameters = parameters
    

    def get_prototypes(self, X, y):
        '''
        X : matriz de covariáveis simbólicas
        y : vetor de rótulos (cada valor varia de 0 até o número de classes - 1)); 

        '''

        prototipos_full = []
        
        for i, n_prots_class_i in enumerate(self.k):  
            
            X_class_i = X[y==i]

            prototipos_class_i = kmeanspp(X_class_i, n_prots_class_i)

            # atualizaçao k-means
            if self.update:
                for t in range(100):
                    D = distance(X_class_i, prototipos_class_i, dist='euclidean')
                    C = np.argmin(D, axis=1)
                    
                    for c in range(n_prots_class_i): 
                        if np.any(C==c):
                            prototipos_class_i[c] = np.mean(X_class_i[C == c], axis = 0)     
             
            prototipos_full.append(prototipos_class_i)  

        # vetor de dimensao sum(k) x 1
        return np.vstack(prototipos_full)
        

    def fit(self, X, y):    

        prototipos = self.get_prototypes(X, y)

        # Matriz de dissimilaridade
        D = distance(X, prototipos, dist=self.dist)
        
        # Matriz de dissimilaridade normalizada
        self.scaler = StandardScaler()
        D_scaled = self.scaler.fit_transform(D)

        # Treinamento do classificador
        clf = self.classifier(**self.parameters)
        clf.fit(D_scaled, y)

        self.clf = clf
        self.prototipos = prototipos

        return


    
    def accuracy(self, X, Y):

        D = distance(X, self.prototipos, self.dist)
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
    # matrix1[0] linhas e matrix2[0] colunas

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
