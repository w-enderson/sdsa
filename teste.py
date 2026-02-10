from models.sdsa import SDSA
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from models.sdsa import distance
from sklearn.preprocessing import StandardScaler


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

dist = 'cityblock'

c = SDSA(**{'dist':dist, 'k':[34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 'classifier': LogisticRegression, 'parameters' : {}})

prots= c.get_prototypes(X_train, y_train)
D_train = distance(X_train, prots, dist=dist)
D_test = distance(X_test, prots, dist=dist)

# pd.DataFrame(D_train).to_csv('D_train.csv', index=False)
# pd.DataFrame(y_train).to_csv('y_train.csv', index=False)

scaler = StandardScaler()
D_train_scaled = scaler.fit_transform(D_train)
D_test_scaled = scaler.transform(D_test)


pca = PCA(n_components=0.999)
D_train_pca = pca.fit_transform(D_train_scaled)
D_test_pca = pca.transform(D_test_scaled)
print(pca.n_components_, D_train.shape)

model1 = LogisticRegression(max_iter=120000)
model1.fit(D_train_scaled, y_train)
print("\nModelo treinado com sucesso!")
y_pred1 = model1.predict(D_test_scaled)
accuracy1 = accuracy_score(y_test, y_pred1)
print(f"\nAcurácia: {accuracy1 * 100:.2f}%")


model2 = LogisticRegression(max_iter=120000)
model2.fit(D_train_pca, y_train)
print("\nModelo treinado com sucesso!")
y_pred2 = model2.predict(D_test_pca)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"\nAcurácia: {accuracy2 * 100:.2f}%")


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from models.sdsa import SDSA, distance

# Configurações
kf = KFold(n_splits=10, shuffle=True, random_state=42)
lista_detalhada = []

print("Iniciando processamento detalhado por fold...")

for i in range(1, 27):
    k_vector = [i] * n_classes
    
    # Loop do K-Fold
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]

        # 1. SDSA
        sdsa = SDSA(**{'dist': dist, 'k': k_vector, 'classifier': LogisticRegression, 'parameters': {}})
        prots = sdsa.get_prototypes(X_train_k, y_train_k)
        
        # 2. Distâncias
        D_train_k = distance(X_train_k, prots, dist=dist)
        D_test_k = distance(X_test_k, prots, dist=dist)
        
        # 3. Normalização
        scaler = StandardScaler()
        D_train_sc = scaler.fit_transform(D_train_k)
        D_test_sc = scaler.transform(D_test_k)
        
        # 4. PCA
        pca = PCA(n_components=0.999)
        D_train_pca = pca.fit_transform(D_train_sc)
        D_test_pca = pca.transform(D_test_sc)
        
        # 5. Modelos e Acurácias
        m_orig = LogisticRegression(max_iter=120000).fit(D_train_sc, y_train_k)
        acc_orig = accuracy_score(y_test_k, m_orig.predict(D_test_sc))
        
        m_pca = LogisticRegression(max_iter=120000).fit(D_train_pca, y_train_k)
        acc_pca = accuracy_score(y_test_k, m_pca.predict(D_test_pca))

        # Salva CADA resultado individualmente
        lista_detalhada.append({
            'k_valor': i,
            'fold': fold_idx + 1,
            'n_prototipos': len(prots),
            'acuracia': acc_orig,
            'modelo': 'Original'
        })
        lista_detalhada.append({
            'k_valor': i,
            'fold': fold_idx + 1,
            'n_prototipos': pca.n_components_,
            'acuracia': acc_pca,
            'modelo': 'PCA_0.999'
        })
    
    print(f"k={i} : acc_o ({acc_orig:2f}) : acc_pca ({acc_pca:2f}) : n_comp ({pca.n_components_})")

# Criar DataFrame
df_detalhado = pd.DataFrame(lista_detalhada)
df_detalhado.to_csv('resultados_completos_folds.csv', index=False)

import seaborn as sns
# --- GRÁFICO BOXPLOT ---
plt.figure(figsize=(16, 8))
sns.boxplot(data=df_detalhado, x='k_valor', y='acuracia', hue='modelo')

plt.title('Distribuição da Acurácia por Fold (Boxplot) para cada k')
plt.xlabel('Número de Protótipos por Classe (k)')
plt.ylabel('Acurácia')
plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# 1. Filtrar pelo modelo específico
df_filtrado = df_detalhado[df_detalhado['modelo'] == 'PCA_0.999']

# 2. Ordenar os valores de K para o eixo X
df_filtrado = df_filtrado.sort_values('k_valor')

# 3. Criar o gráfico de barras
plt.figure(figsize=(10, 6))

# O seaborn calcula a média automaticamente e adiciona o intervalo de confiança (ci)
# 'capsize' adiciona as abas no topo da barra de erro
sns.barplot(
    data=df_filtrado, 
    x='k_valor', 
    y='n_prototipos', 
    capsize=.1, 
    palette='viridis'
)

plt.title('Média de n_prototipos por k_valor (Modelo: PCA_0.999)')
plt.xlabel('Valor de K')
plt.ylabel('Média de n_prototipos (com Erro Padrão)')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()






# 3. Configurar o gráfico
plt.figure(figsize=(12, 7))

# Boxplot de Acurácia
sns.boxplot(
    x='k_valor', 
    y='acuracia',       # Alterado para acurácia
    hue='modelo',      # Compara Original vs PCA
    data=df_detalhado, 
    palette='Set1',    # Usei Set1 para cores mais vibrantes (Original vs PCA)
    showfliers=False
)

# Jitter (Stripplot) com dodge=True para alinhar com os boxplots
sns.stripplot(
    x='k_valor', 
    y='acuracia', 
    hue='modelo', 
    data=df_detalhado, 
    dodge=True,        # Alinha os pontos em cima de cada caixa lateral
    color='black', 
    alpha=0.3, 
    jitter=0.2, 
    size=5,
    legend=False       # Não duplica a legenda
)

plt.title('Comparação de Acurácia: Original vs PCA_0.999 (10-Folds)')
plt.xlabel('Valor de k (Vizinhos)')
plt.ylabel('Acurácia')
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Ajuste fino da escala do eixo Y (opcional)
# plt.ylim(0.8, 1.0) # Descomente se quiser focar em uma faixa específica

plt.legend(title='Modelo', loc='lower right')
plt.tight_layout()
plt.show()





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Pivotar o DataFrame para ter Original e PCA na mesma linha por Fold e K
# Isso facilita subtrair um do outro
df_pivot = df_detalhado.pivot(index=['k_valor', 'fold'], columns='modelo', values='n_prototipos').reset_index()

# 2. Calcular a diferença (Economia de parâmetros)
# Diferença = Original - PCA_0.999
df_pivot['diferenca_n'] = df_pivot['Original'] - df_pivot['PCA_0.999']

# 3. Ordenar para o gráfico
df_pivot = df_pivot.sort_values('k_valor')

# 4. Criar o gráfico de barras da diferença média
plt.figure(figsize=(10, 6))

sns.barplot(
    data=df_pivot, 
    x='k_valor', 
    y='diferenca_n', 
    capsize=.1, 
    color='salmon'
)

# Adicionar Jitter por cima para ver a variação da economia entre os folds
sns.stripplot(
    data=df_pivot, 
    x='k_valor', 
    y='diferenca_n', 
    color='black', 
    alpha=0.3, 
    jitter=True
)

plt.title('Redução Média no Nº de Protótipos (Original - PCA)\nQuanto maior a barra, maior a economia de espaço')
plt.xlabel('Valor de K')
plt.ylabel('Diferença (Nº de protótipos reduzidos)')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()