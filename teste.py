from models.sdsa import SDSA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from models.sdsa import distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 

data= pd.read_csv('datasets/climates.csv')

X = data.drop('target', axis=1).values                                                                                                                                         
y = data['target'].values

n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=10,
    stratify=y
)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
dist = 'euclidean'
c = SDSA(**{'dist':dist, 'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': LogisticRegression, 'parameters' : {}})

prots= c.get_prototypes(X_train, y_train)
D = distance(X_train, prots, dist=dist)
D2= distance(X_test, prots, dist=dist)

scaler = StandardScaler()
D_scaled = scaler.fit_transform(D)
D2_scaled = scaler.transform(D2)

model = LogisticRegression(max_iter=120000)
model.fit(D_scaled, y_train)
print("\nModelo treinado com sucesso!")

y_pred = model.predict(D2_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {accuracy * 100:.2f}%")

# 2. Relatório de Classificação Completo
# Mostra precisão, recall, f1-score para cada classe
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 3. Matriz de Confusão
# Mostra quantos previu certo e quantos errou, e onde errou
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))



# start = time.time()
# c.fit(X_train, y_train)
# end = time.time()


# acc = c.accuracy(x_test, y_test)
# results.append(
#     ["climates", n_classes, X.shape[1]/2, X.shape[0], "sdsa_lr", 10,
#     fold_id, acc]
# )

# fold_id += 1


