from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import time
import os


def rodar_experimentos(classifier_name, classifier, dataset, n_splits=10, mc_range=range(10)):
    columns = ['dataset', 'n_classes', 'n_features', 'n_samples', 
               'classifier', 'mc', 'test_fold', 'acc', 'exec_time']
    results = []

    # Carregar dataset
    data = pd.read_csv(f'./datasets/{dataset}.csv')
    X = data.drop('target', axis=1).values
    y = data['target'].values
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    n_samples = X.shape[0]

    # Loop pelos seeds (Monte Carlo)
    for mc in mc_range:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=mc)

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f'Computing: dataset={dataset}, classifier={classifier_name}, mc={mc}, fold={fold_id}')
            x_train, y_train = X[train_idx], y[train_idx]
            x_test, y_test = X[test_idx], y[test_idx]

            c = classifier()
            start = time.time()
            c.fit(x_train, y_train)
            end = time.time()

            exec_time = end - start
            predicoes = c.predict(x_test)
            acc = np.mean(predicoes == y_test)

            results.append([
                dataset, n_classes, n_features, n_samples, classifier_name, 
                mc, fold_id, acc, exec_time
            ])

    df = pd.DataFrame(results, columns=columns)
    return df


# ------------------------------------------------------
# Lista de classificadores
# ------------------------------------------------------
classificadores = {
    "DecisionTree": DecisionTreeClassifier,
    "SVM": SVC,
    "LogisticRegression": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "XGBoost": XGBClassifier,
    "RandomForest": RandomForestClassifier,
}

# ------------------------------------------------------
# Lista de datasets
# ------------------------------------------------------
datasets = ["mushroom", "climates", "dry-climates", "european-climates"]

# ------------------------------------------------------
# Rodar todos os classificadores em todos os datasets
# ------------------------------------------------------
todos_resultados = []

for dataset in datasets:
    for nome, modelo in classificadores.items():
        df_temp = rodar_experimentos(nome, modelo, dataset)
        todos_resultados.append(df_temp)

# Unir tudo em um único DataFrame
df_todos = pd.concat(todos_resultados, ignore_index=True)

# ------------------------------------------------------
# Criar resumo (médias e desvios padrão por dataset e classificador)
# ------------------------------------------------------
df_resumo = (
    df_todos.groupby(["dataset", "classifier"])["acc"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "acc_mean", "std": "acc_std"})
)

# ------------------------------------------------------
# Salvar resultados em CSV
# ------------------------------------------------------
os.makedirs("", exist_ok=True)

df_todos.to_csv("results_no_sdsa/resultados_gerais.csv", index=False)
df_resumo.to_csv("results_no_sdsa/resultados_resumidos.csv", index=False)

print("\nResumo de desempenho por dataset e classificador:\n")
print(df_resumo)

print("\nArquivos salvos em: ./results_no_sdsa/")
print(" - resultados_gerais.csv (todas as execuções)")
print(" - resultados_resumidos.csv (médias e desvios padrão)")
