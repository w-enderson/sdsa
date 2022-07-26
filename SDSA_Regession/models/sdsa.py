import scipy.sparse as sp
import random
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import stable_cumsum
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 

class CenterRangeRegression:
    def __init__(self):
        self.center_model = None
        self.range_model = None
    def fit(self, D, y):
        pass

    def predict(self, D):
        pred_centers = self.center_model.predict(D)
        pred_rangers = self.range_model.predict(D)
        y_min = np.array(pred_centers) - [pred_rangers/2]
        y_max = np.array(pred_centers) + [pred_rangers/2]
        return y_min, y_max


def transform_center_range(y):
    center_model = np.sum(y, axis = 1)/2
    range_model = np.diff(y).ravel()
    return center_model, range_model


class CenterRangeSVR(CenterRangeRegression):
    def fit(self, D, y):
       center, ranger = transform_center_range(y)
       self.center_model = SVR().fit(D, center)
       self.range_model = SVR().fit(D, ranger)
       return self

class CenterRangeRF(CenterRangeRegression):
    def fit(self, D, y):
       center, ranger = transform_center_range(y)
       self.center_model = RandomForestRegressor().fit(D, center)
       self.range_model = RandomForestRegressor().fit(D, ranger)
       return self

class CenterRangeLinear(CenterRangeRegression):
    def fit(self, D, y):
       center, ranger = transform_center_range(y)
       self.center_model = LinearRegression().fit(D, center)
       self.range_model = LinearRegression().fit(D, ranger)
       return self

class CenterRangeLinearComparition():
    def __init__(self):
        self.center_model = None
        self.range_model = None
    def fit(self, X_centers, X_rangers, y):
       center, ranger = transform_center_range(y)

       self.center_model = LinearRegression().fit(X_centers, center)
       self.range_model = LinearRegression().fit(X_rangers, ranger)
       return self
    def predict(self, X_centers, X_rangers):
        pred_centers = self.center_model.predict(X_centers)
        pred_rangers = self.range_model.predict(X_rangers)
        y_min = np.array(pred_centers) - [pred_rangers/2]
        y_max = np.array(pred_centers) + [pred_rangers/2]
        return y_min, y_max

class CenterRangeOLS():
    def __init__(self):
        self.ols_model = None
    def fit(self, D, y):
       center, ranger = transform_center_range(y)
       self.ols_model = sm.OLS(pd.DataFrame([center, ranger]).T, D).fit()
       return self
    def predict(self, D):
        pred_ols = self.ols_model.predict(D)
        y_min = np.array(pred_ols[:,0]) - [pred_ols[:,1]/2]
        y_max = np.array(pred_ols[:,0]) + [pred_ols[:,1]/2]
        return y_min, y_max


class SDSR:

    def __init__ (self, k, update = True, classifier = CenterRangeSVR, parameters = None):
        self.k = k
        self.update = update
        self.classifier = classifier
        self.parameters = parameters
    

    def fit(self, X, y):    
        
        #print(X)
        #print(X[:,0:2])
        if self.classifier == CenterRangeLinearComparition:

                X_centers, X_rangers = transform_X_center_ranger(X)
             
                rm = self.classifier(**self.parameters)

                rm.fit(X_centers, X_rangers, y) # y matriz com duas colunas (min e max)

                self.rm = rm
        
        else:

            media_prot = []

            medias = kmeanspp(X, self.k)

            if self.update == True:
                for t in range(100):
                    D = distances(X, medias)
                    C  = np.argmin(D, axis=1)

            media_prot.append(medias)  

            prots = np.vstack(media_prot)

            D = distances(X, prots)

            rm = self.classifier(**self.parameters)

            rm.fit(D, y) # y matriz com duas colunas (min e max)

            self.rm = rm
            self.medias = prots
        return self
  
     
    # def accuracy(self, X, Y):

    #     if self.classifier == CenterRangeLinearComparition:
    #         X_centers, X_rangers = transform_X_center_ranger(X)
    #         predicoes = self.clf.predict(X_centers, X_rangers)
    #     else:
    #         D = distances(X, self.medias)
    #         predicoes = self.clf.predict(D)
        
    #     accuracy = np.sum(predicoes == Y)/len(predicoes == Y)
    #     return accuracy    
    
    
    # def r_square(self, y, rm):
    #     SST = np.var(np.diff(y))
    #     SSReg = np.var(np.diff(rm))
    #     Rsquared = SSReg/SST

    #     return Rsquared
    
    def mmre(self, X, Y):

        if self.classifier == CenterRangeLinearComparition:
            X_centers, X_rangers = transform_X_center_ranger(X)
            y_predict = self.rm.predict(X_centers, X_rangers)
        else:
            D = distances(X, self.medias)
            y_predict = self.rm.predict(D)

        mmre = np.sum(((Y[:,0] - y_predict[0])**2 + (Y[:,1] - y_predict[1])**2))/(len(Y)*2)
        return mmre


def transform_X_center_ranger(X):

    X_centers = pd.DataFrame()
    X_rangers = pd.DataFrame()

    for i, j in zip(range(0,X.shape[1],2), range(int(X.shape[1]/2))):
        
        X_centers['x_{}'.format(j)] =  transform_center_range(X[:,i:i+2])[0]
        X_rangers['x_{}'.format(j)] =  transform_center_range(X[:,i:i+2])[1]
    
    return X_centers, X_rangers

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
