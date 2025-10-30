from __future__ import division

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix


class Dataset:
    def __init__(self, dataframe=None, data=None, target=None):
        if dataframe is not None:
            data_columns = dataframe.columns[dataframe.columns != "label"]
            self.data = dataframe[data_columns].values
            self.target = dataframe["label"].values
        else:
            self.data = data
            self.target = target

def plot_pontos(sample):
    plt.figure(figsize=(8, 6))
    plt.scatter(sample[:, 0], sample[:, 1], c='blue', alpha=0.5)
    plt.title("Amostra de uma Normal Multivariada 2D")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("grafico.png")  # Salva o gráfico em um arquivo
    plt.close()


def plot_pontos_com_retangulo(sample, inf, sup):
    plt.figure(figsize=(8, 6))

    # Pontos da amostra
    plt.scatter(sample[:, 0], sample[:, 1], c='blue', alpha=0.5, label='Amostra')

    # Pontos inferior (min) e superior (max)
    plt.scatter(*inf, c='red', s=100, label='Inferior (mínimo)')
    plt.scatter(*sup, c='green', s=100, label='Superior (máximo)')

    # Retângulo delimitador
    width = sup[0] - inf[0]
    height = sup[1] - inf[1]
    rect = plt.Rectangle(inf, width, height, fill=False, color='orange', lw=2, label='Retângulo')
    plt.gca().add_patch(rect)

    # Aparência
    plt.title("Amostra com Retângulo (inf/sup)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("grafico2.png")  # Salva o gráfico em um arquivo
    plt.close()


def generate_multivariate_gaussians(parameters):
    registros = []  # ← onde vamos armazenar cada linha do dataset

    n_gaussians = len(parameters)
    print(f"Número de gaussianas: {n_gaussians}")

    for gaussian in range(n_gaussians):
        mu = np.array(parameters[gaussian]['mu'])
        sigma = np.array(parameters[gaussian]['sigma'])
        sample_size = parameters[gaussian]['quantity']
        label = parameters[gaussian]['label']

        dim = len(mu)  # número de variáveis (dimensão)
        print(f"\nGerando {sample_size} amostras {dim}D para label {label}")

        # gera amostra principal
        sample = np.random.multivariate_normal(mu, sigma, sample_size)
        plot_pontos(sample)

        for example in sample:
            # gera subamostra ao redor do ponto "example"
            n = np.random.randint(10, 50)
            medias = example
            covariancias = make_spd_matrix(n_dim=dim) * 8
            sample2 = np.random.multivariate_normal(medias, covariancias, n)

            inf = np.min(sample2, axis=0)
            sup = np.max(sample2, axis=0)

            # cria registro dinâmico
            registro = {}
            for i in range(dim):
                registro[f'x{i+1}_min'] = inf[i]
                registro[f'x{i+1}_max'] = sup[i]
            registro['target'] = label

            registros.append(registro)

            # opcional: plotar retângulo se for 2D
            # plot_pontos_com_retangulo(sample2, inf, sup)

    # cria DataFrame com colunas ordenadas dinamicamente
    colunas = [f'x{i+1}_{suf}' for i in range(dim) for suf in ('min', 'max')] + ['target']
    df = pd.DataFrame(registros, columns=colunas)

    print("\nDataset criado com shape:", df.shape)
    print(df.head())

    return df


parameters_dict = {
    "mu": np.array([99, 99]),
    "sigma": np.array([
                [200, 0],
                [0, 100]
            ]),
    "quantity": 200,
    "label": 0
}
parameters_dict2 = {
    "mu": np.array([125, 80]),
    "sigma": np.array([
                [100, 0],
                [0, 200]
            ]),
    "quantity": 200,
    "label": 1
}
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_retangulos(df, alpha=0.3, show_points=True, color_map=None):
    """
    Plota os retângulos (intervalos) de um dataset 2D no formato:
    x1inf, x1sup, x2inf, x2sup, label

    Parâmetros:
    - df: DataFrame com colunas x1inf, x1sup, x2inf, x2sup, label
    - alpha: transparência dos retângulos
    - show_points: se True, plota também os centros
    - color_map: dicionário opcional {label: cor}
    """
    # Verificação de dimensões
    required_cols = ['x1_min', 'x1_max', 'x2_min', 'x2_max', 'target']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada no dataset")

    # Define cores por classe
    labels = sorted(df['target'].unique())
    if color_map is None:
        cmap = matplotlib.colormaps.get_cmap('tab10')
        color_map = {label: cmap(i) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        x_inf, x_sup = row['x1_min'], row['x1_max']
        y_inf, y_sup = row['x2_min'], row['x2_max']
        label = row['target']

        # Calcula dimensões
        width = x_sup - x_inf
        height = y_sup - y_inf
        color = color_map[label]

        # Retângulo
        rect = patches.Rectangle((x_inf, y_inf), width, height,
                                 linewidth=1.5, edgecolor=color, facecolor=color,
                                 alpha=alpha, label=f'Classe {label}')
        ax.add_patch(rect)

        # Centro do retângulo
        if show_points:
            cx = (x_inf + x_sup) / 2
            cy = (y_inf + y_sup) / 2
            ax.scatter(cx, cy, c=[color], s=30, edgecolors='k')

    # Formatação do gráfico
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Retângulos gerados por amostras (intervalos)')
    ax.grid(True)
    ax.axis('equal')

    # Legenda sem duplicatas
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.savefig("grafico3.png")  # Salva o gráfico em um arquivo
    plt.close()


dataset = generate_multivariate_gaussians([parameters_dict, parameters_dict2])
dataset.to_csv("dataset5.csv", index=False)

plot_retangulos(dataset)

