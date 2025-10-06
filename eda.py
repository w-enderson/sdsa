import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("datasets/climates.csv")
data

X = data.drop(columns="target")
y = data["target"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Primeiro o split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Agora escalonamos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)  

# Treinando o modelo
model = LogisticRegression(max_iter=120000)
model.fit(X_train_scaled, y_train)

# Predição
y_pred = model.predict(X_test_scaled)

# Avaliando
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Matriz de Confusão:\n{conf_matrix}')
