from models.sdsa import SDSA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
c = SDSA(**{'k': [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': LogisticRegression, 'parameters' : {'max_iter' : 120000}})

prots= c.get_prototipes(X_train, y_train)
D = distance(X, prots, dist="euclidean")

# min_distancias = D.min(axis=0)
# max_distancias = D.max(axis=0)
# plt.figure(figsize=(8, 6))
# plt.scatter(min_distancias, max_distancias, alpha=0.7)

# means = X_train.mean(axis=0)
# stds = X_train.std(axis=0)
# plt.figure(figsize=(8, 6))
# plt.scatter(means, stds, alpha=0.7)

print(D.max(axis=1))
print(D.min(axis=1))

print(D.shape)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)
model.fit(X_train, y_train)
print("\nModelo treinado com sucesso!")


y_pred = model.predict(X_test_scaled)
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


