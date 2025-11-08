from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.datasets import make_spd_matrix


# =====================================================
# Classe Dataset (opcional, mas mantida por clareza)
# =====================================================
class Dataset:
    def __init__(self, dataframe=None, data=None, target=None):
        if dataframe is not None:
            data_columns = dataframe.columns[dataframe.columns != "label"]
            self.data = dataframe[data_columns].values
            self.target = dataframe["label"].values
        else:
            self.data = data
            self.target = target


# =====================================================
# Funções de Plotagem
# =====================================================
def plot_pontos(sample):
    plt.figure(figsize=(8, 6))
    plt.scatter(sample[:, 0], sample[:, 1], c='blue', alpha=0.5)
    plt.title("Amostra de uma Normal Multivariada 2D")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("grafico.png")
    plt.close()


def plot_retangulos(df, alpha=0.3, show_points=True, color_map=None):
    """
    Plota os retângulos (intervalos) de um dataset 2D:
    colunas x1_min, x1_max, x2_min, x2_max, target
    """
    required_cols = ['x1_min', 'x1_max', 'x2_min', 'x2_max', 'target']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada no dataset")

    # Define cores
    labels = sorted(df['target'].unique())
    if color_map is None:
        cmap = plt.colormaps.get_cmap('tab10')
        color_map = {label: cmap(i) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        x_inf, x_sup = row['x1_min'], row['x1_max']
        y_inf, y_sup = row['x2_min'], row['x2_max']
        label = row['target']

        width, height = x_sup - x_inf, y_sup - y_inf
        color = color_map[label]

        rect = patches.Rectangle((x_inf, y_inf), width, height,
                                 linewidth=1.5, edgecolor=color, facecolor=color,
                                 alpha=alpha, label=f'Classe {label}')
        ax.add_patch(rect)

        if show_points:
            cx = (x_inf + x_sup) / 2
            cy = (y_inf + y_sup) / 2
            ax.scatter(cx, cy, c=[color], s=30, edgecolors='k')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Retângulos gerados por amostras (intervalos)')
    ax.grid(True)
    ax.axis('equal')

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.savefig("./dataset5-plot.png")
    plt.close()


# =====================================================
# Geração de Gaussianas e Dataset
# =====================================================
def generate_multivariate_gaussians(parameters):
    registros = []
    n_gaussians = len(parameters)
    print(f"Número de gaussianas: {n_gaussians}")

    for gaussian in range(n_gaussians):
        mu = np.array(parameters[gaussian]['mu'])
        sigma = np.array(parameters[gaussian]['sigma'])

        sample_size = parameters[gaussian]['quantity']
        label = parameters[gaussian]['label']

        dim = len(mu)
        print(f"\nGerando {sample_size} amostras {dim}D para label {label}")

        sample = np.random.multivariate_normal(mu, sigma, sample_size)
        plot_pontos(sample)
        cov = make_spd_matrix(n_dim=dim) * 8

        for example in sample:
            n = np.random.randint(10, 50)
            sample2 = np.random.multivariate_normal(example, cov, n)

            inf = np.min(sample2, axis=0)
            sup = np.max(sample2, axis=0)

            registro = {f'x{i+1}_min': inf[i] for i in range(dim)}
            registro.update({f'x{i+1}_max': sup[i] for i in range(dim)})
            registro['target'] = label
            registros.append(registro)

    colunas = [f'x{i+1}_{suf}' for i in range(dim) for suf in ('min', 'max')] + ['target']
    df = pd.DataFrame(registros, columns=colunas)

    print("\nDataset criado com shape:", df.shape)
    print(df.head())
    return df


# =====================================================
# Execução principal
# =====================================================
if __name__ == "__main__":

    # dataset5
    dataset5_dict1 = {
        "mu": np.array([99, 99]),
        "sigma": np.array([[200, 0],
                           [0, 100]]),
        "quantity": 200,
        "label": 0
    }

    dataset5_dict2 = {
        "mu": np.array([125, 80]),
        "sigma": np.array([[100, 0],
                           [0, 200]]),
        "quantity": 200,
        "label": 1
    }

    dataset5 = generate_multivariate_gaussians([dataset5_dict1, dataset5_dict2])
    dataset5.to_csv("./dataset5.csv", index=False)
    plot_retangulos(dataset5)

    # dataset6

