import os
import pandas as pd
import glob

# Pasta raiz dos resultados
root_path = 'results_test'

# Classificadores
classifiers = [
    'ivabc', 'knn'
]

rows = {}

for classifier in classifiers:
    path = os.path.join(root_path, classifier, '*.csv')
    csv_files = glob.glob(path)

    for f in csv_files:
        nome = os.path.basename(f).replace('.csv','')
        parts = nome.split('-')
        dataset = parts[1]  # ✅ dataset-x-mc-y.csv → dataset está na posição 0

        df = pd.read_csv(f)

        # Se houver coluna classifier, filtra
        if 'classifier' in df.columns:
            df = df[df['classifier'] == classifier]

        # Calcula estatísticas
        stats = df[['acc', 'exec_time']].agg(['mean','std'])

        # Cria linha se não existir
        if dataset not in rows:
            rows[dataset] = {'dataset': dataset}

        # Nomes das colunas
        acc_mean_col = f'{classifier}_acc_mean'
        acc_std_col = f'{classifier}_acc_std'
        time_mean_col = f'{classifier}_time_mean'
        time_std_col = f'{classifier}_time_std'

        rows[dataset][acc_mean_col] = stats.loc['mean', 'acc']
        rows[dataset][acc_std_col] = stats.loc['std', 'acc']
        rows[dataset][time_mean_col] = stats.loc['mean', 'exec_time']
        rows[dataset][time_std_col] = stats.loc['std', 'exec_time']

# Cria DataFrame final
df_final = pd.DataFrame(rows.values())

# ✅ Ordenação das colunas na ordem desejada
ordered_cols = ['dataset',  'ivabc_acc_mean', 'knn_acc_mean', 
                            'ivabc_acc_std', 'knn_acc_std', 
                            'ivabc_time_mean', 'knn_time_mean', 
                            'ivabc_time_std', 'knn_time_std',
                            ]

# Ordem específica:
metrics = [
    'acc_mean',   # todas as médias de acurácia primeiro
    'acc_std',    # depois todos os desvios de acurácia
    'time_mean',
    'time_std'
]

for metric in metrics:
    for clf in classifiers:
        col = f'{metric}_{clf}'
        if col in df_final.columns:
            ordered_cols.append(col)

df_final = df_final[ordered_cols].round(4)


# Salva CSV final
os.makedirs('./agregacaoo', exist_ok=True)
df_final.to_csv('./agregacaoo/summary_results_not_sdsa.csv', index=False)

print("CSV final gerado: summary_results_not_sdsa.csv ✅")
