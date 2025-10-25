import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from util.functions import generate_multivariate_gaussians
import numpy as np

# Gerando dados
parameters = np.array([
    [99, 9, 99, 169, 200, 0],
    [108, 9, 99, 169, 200, 1]
])
data = generate_multivariate_gaussians(parameters, 10)

# Ajuste: pegar X e y do objeto retornado
X = data.data       # array de features
y = data.target.astype(int)  # targets

# Criando DataFrame
df = pd.DataFrame(X, columns=["x1_min", "x1_max", "x2_min", "x2_max"])
df['target'] = y

# Definindo cores por target
colors = {0: 'red', 1: 'green', 2: 'blue'}

# Plot
plt.figure(figsize=(10,6))
ax = plt.gca()

for _, row in df.iterrows():
    rect = Rectangle(
        (row['x1_min'], row['x2_min']),         # canto inferior esquerdo
        row['x1_max'] - row['x1_min'],          # largura
        row['x2_max'] - row['x2_min'],          # altura
        edgecolor=colors.get(row['target'], 'black'),
        facecolor='none',                        # sem preenchimento
        linewidth=1
    )
    ax.add_patch(rect)
    ax.set_xlim(df['x1_min'].min() - 5, df['x1_max'].max() + 5)
    ax.set_ylim(df['x2_min'].min() - 5, df['x2_max'].max() + 5)


plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Retângulos representando intervalos de x1 e x2')
plt.grid(True)
plt.show()
