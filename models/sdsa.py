
from sklearn.datasets import load_iris
import random
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

class SDSA:

    def __init__ (self, k):
        self.k = k

    def fit(self, X, Y):    
        #treinamento
        sort = random.sample(range(len(X)), self.k)
        #print(sort)
        n = X[sort]
        #print(n)

        distance = cdist(X, n)
        C  = np.argmin(distance, axis=1)
        #print(C)

        medias = []
        for c in range(self.k): 
            medias.append(np.mean(X[(C == c)], axis = 0))

        medias = np.array(medias)
        medias_min = medias[:,::2]
        medias_max = medias[:,1::2]
        #print(medias)

        for i in range(100):

            d_min = cdist(X[:,::2], medias_min)
            d_max = cdist(X[:,1::2], medias_max)    
            distance = d_min + d_max
            C  = np.argmin(distance, axis=1)

            #print(distance)
            #print(C)
            minimos = np.min(distance, axis=1)
            #print(minimos)

            medias = []
            for c in range(self.k): 
                medias.append(np.mean(X[(C == c)], axis = 0))
                
        medias = np.array(medias)
        medias_min = medias[:,::2]
        medias_max = medias[:,1::2]
        d_min = cdist(X[:,::2], medias_min)
        d_max = cdist(X[:,1::2], medias_max)


        D = d_min + d_max

        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(D,Y)
        self.clf = clf
        self.medias = medias
        return self

    def accuracy(self, X, Y):
        #teste e retorna acuracia
        #predições para o teste

        D_min = cdist(X[:, ::2], self.medias[:,::2])
        D_max = cdist(X[:, 1::2], self.medias[:,1::2])

        D = D_min + D_max
        predicoes = self.clf.predict(D)
        
        accuracy = np.mean(predicoes == Y)
        return accuracy    





