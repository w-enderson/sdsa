
from sklearn.datasets import load_iris
import random
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

class Modelo:

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
        #print(medias)

        for i in range(100):

            distance = cdist(X, medias)
            C  = np.argmin(distance, axis=1)

            #print(distance)
            #print(C)
            minimos = np.min(distance, axis=1)
            #print(minimos)

            medias = []
            for c in range(self.k): 
                medias.append(np.mean(X[(C == c)], axis = 0))

        d_min = cdist(X.iloc[:,0:len(X.columns)-1:2], medias)
        d_max = cdist(X.iloc[:,1:len(X.columns)-1:2], medias)


        D = sum(d_min, d_max)

        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(D,Y)
        self.clf = clf
        self.medias = medias
        return self

    def acuracia(self, X, Y):
        #teste e retorna acuracia
        #predições para o teste
        D = cdist(X, self.medias)
        predicoes = self.clf.predict(D)
        
        acuracia = np.sum(predicoes == Y)/len(predicoes)
        return acuracia    

class Simulacao:
    def __init__ (self, n):
        self.n = n
    #funcao de montecarlo 
    def montecarlo(self):
        np.random.seed(42)
        resultados = []
        for _ in range(self.n): 
            #np.random.seed(mc)
            indices_emb = np.random.choice(150, 150, replace = False)
            dados_embaralhados = iris['data'][indices_emb]
            target_embaralhados = iris['target'][indices_emb]
            skf = StratifiedKFold(n_splits=5) #dividindo em 5 grupos para ter como treino e teste
            #print(skf)
            #validação cruzada
            for train_index, test_index in skf.split(dados_embaralhados, target_embaralhados):
                treino = {
                    'data': dados_embaralhados[train_index],
                    'target': target_embaralhados[train_index]
                }
                teste = {
                    'data': dados_embaralhados[test_index],
                    'target': target_embaralhados[test_index]
                }

                n = Modelo(k = 4)
                n.fit(X = treino['data'], Y = treino['target'])
                acc = n.acuracia( X = teste['data'], Y = teste['target'])
                resultados.append(acc)

        return resultados


print(Simulacao(n = 5).montecarlo())
#print(type(iris['data']))




