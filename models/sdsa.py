
from sklearn.datasets import load_iris
import random
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class SDSA:

    def __init__ (self, k, update = True, classifier = DecisionTreeClassifier):
        self.k = k
        self.update = update
        self.classifier = classifier

    def fit(self, X, Y):    
        #treinamento
        sort = np.random.choice(range(len(X)), self.k)
        #print(sort)
        n = X[sort]
        #print(n)

        distance = cdist(X, n)
        C  = np.argmin(distance, axis=1)
        #print(C)

        medias = []
        for c in range(self.k): 
            if np.any(C==c):
                medias.append(np.mean(X[(C == c)], axis = 0))

        medias = np.array(medias)
        medias_min = medias[:,::2]
        medias_max = medias[:,1::2]
        #print(medias)

        if self.update == True:
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
                    if np.any(C==c):
                      medias.append(np.mean(X[(C == c)], axis = 0))
                    
                medias = np.array(medias)
                medias_min = medias[:,::2]
                medias_max = medias[:,1::2]
                d_min = cdist(X[:,::2], medias_min)
                d_max = cdist(X[:,1::2], medias_max)
                D = d_min + d_max
        
        d_min = cdist(X[:,::2], medias_min)
        d_max = cdist(X[:,1::2], medias_max)
        D = d_min + d_max

        clf = self.classifier()
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
        
        accuracy = np.sum(predicoes == Y)/len(predicoes == Y)
        return accuracy    





