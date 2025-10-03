import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches

for i in [1,2,4]:
        
    # Carregar os dados
    df = pd.read_csv(f"synthetic_datasets/synthetic-dataset{i}.csv")

    # Criar figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotar retângulos por linha
    for _, row in df.iterrows():
        width = row['x1_max'] - row['x1_min']
        height = row['x2_max'] - row['x2_min']
        color = 'cyan' if row['target'] == 0 else 'yellow'
        rect = patches.Rectangle(
            (row['x1_min'], row['x2_min']),
            width,
            height,
            edgecolor="black",
            facecolor=color,
            alpha=0.5
        )
        ax.add_patch(rect)

    # 🧠 Atualizar limites dos eixos com base no dataset
    ax.set_xlim(df['x1_min'].min(), df['x1_max'].max())
    ax.set_ylim(df['x2_min'].min(), df['x2_max'].max())

    # Legenda
    legend_elements = [
        patches.Patch(facecolor='cyan', edgecolor='cyan', alpha=0.3, label='Classe 0'),
        patches.Patch(facecolor='yellow', edgecolor='yellow', alpha=0.3, label='Classe 1')
    ]
    ax.legend(handles=legend_elements)

    # Configurações finais
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Visualização do dataset{i} por Classe")
    ax.grid(True)
    plt.tight_layout()

    # Salvar imagem
    plt.savefig(f"./plots/dataset{i}-plot.png")
    print("Salvo")






import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Carregar dataset 3
df = pd.read_csv("synthetic_datasets/synthetic-dataset3.csv")

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

def cuboid_data(x_min, x_max, y_min, y_max, z_min, z_max):
    # Define os 8 vértices do cubóide
    return [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ]

def plot_cuboid(ax, vertices, color):
    # Define as 6 faces do cubóide usando os vértices
    faces = [
        [vertices[j] for j in [0,1,2,3]],  # bottom
        [vertices[j] for j in [4,5,6,7]],  # top
        [vertices[j] for j in [0,1,5,4]],  # front
        [vertices[j] for j in [2,3,7,6]],  # back
        [vertices[j] for j in [1,2,6,5]],  # right
        [vertices[j] for j in [0,3,7,4]]   # left
    ]
    poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=0.4)
    ax.add_collection3d(poly3d)

# Plotar cada intervalo como cubóide
for _, row in df.iterrows():
    vertices = cuboid_data(
        row['x1_min'], row['x1_max'],
        row['x2_min'], row['x2_max'],
        row['x3_min'], row['x3_max']
    )
    color = 'cyan' if row['target'] == 0 else 'yellow'
    plot_cuboid(ax, vertices, color)

# Ajustar limites dos eixos
ax.set_xlim(df['x1_min'].min(), df['x1_max'].max())
ax.set_ylim(df['x2_min'].min(), df['x2_max'].max())
ax.set_zlim(df['x3_min'].min(), df['x3_max'].max())

# Labels e título
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Visualização 3D dos Intervalos - Dataset 3')

# Legenda manual
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='cyan', edgecolor='k', label='Classe 0', alpha=0.4),
    Patch(facecolor='yellow', edgecolor='k', label='Classe 1', alpha=0.4)
]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig("./plots/dataset3-3d-plot.png")
print("Gráfico 3D salvo como 'dataset3-3d-plot.png'")
