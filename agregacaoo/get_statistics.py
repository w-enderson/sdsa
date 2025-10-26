import os
import pandas as pd
import glob

# Pasta raiz dos resultados
root_path = 'results_test'

# Pega todos os classificadores (pastas dentro de results_test)
classifiers = ['sdsa', 'sdsa_knn']

# Distâncias que vamos considerar
distances = ['euclidean', 'sqeuclidean', 'cityblock', 'hausdorff']

rows = []

for classifier in classifiers:
    path = os.path.join(root_path, classifier, '*.csv')
    csv_files = glob.glob(path)
    
    # Agrupa arquivos por (dataset, distancia)
    grouped_files = {}
    for f in csv_files:
        nome = os.path.basename(f).replace('.csv','')
        parts = nome.split('-')
        dataset = parts[1]
        distancia = parts[3]
        key = (dataset, distancia)
        grouped_files.setdefault(key, []).append(f)
    
    # Para cada grupo, concatena e calcula estatísticas
    for (dataset, distancia), file_list in grouped_files.items():
        dfs = [pd.read_csv(f) for f in file_list]
        df_total = pd.concat(dfs, ignore_index=True)
        
        # Se houver coluna classifier, filtra
        if 'classifier' in df_total.columns:
            df_total = df_total[df_total['classifier'] == classifier]
        
        stats = df_total[['acc', 'exec_time']].agg(['mean','std'])
        
        # Procura linha existente para dataset + classificador
        existing = next((row for row in rows if row['dataset']==dataset and row['classifier']==classifier), None)
        
        col_prefix = distancia
        acc_mean_col = f'{col_prefix}_acc_mean'
        acc_std_col = f'{col_prefix}_acc_std'
        time_mean_col = f'{col_prefix}_time_mean'
        time_std_col = f'{col_prefix}_time_std'
        
        if existing is None:
            row = {
                'dataset': dataset,
                'classifier': classifier,
                acc_mean_col: stats.loc['mean','acc'],
                acc_std_col: stats.loc['std','acc'],
                time_mean_col: stats.loc['mean','exec_time'],
                time_std_col: stats.loc['std','exec_time']
            }
            rows.append(row)
        else:
            existing[acc_mean_col] = stats.loc['mean','acc']
            existing[acc_std_col] = stats.loc['std','acc']
            existing[time_mean_col] = stats.loc['mean','exec_time']
            existing[time_std_col] = stats.loc['std','exec_time']

# Cria DataFrame final
df_final = pd.DataFrame(rows)

# Reordena colunas: primeiro dataset e classifier
cols = ['dataset', 'classifier']

# 1️⃣ Todas as médias de acurácia
for suffix in ['acc_mean']:
    for dist in distances:
        col_name = f'{dist}_{suffix}'
        if col_name in df_final.columns:
            cols.append(col_name)

# 2️⃣ Todas as std de acurácia
for suffix in ['acc_std']:
    for dist in distances:
        col_name = f'{dist}_{suffix}'
        if col_name in df_final.columns:
            cols.append(col_name)

# 3️⃣ Todas as médias de tempo
for suffix in ['time_mean']:
    for dist in distances:
        col_name = f'{dist}_{suffix}'
        if col_name in df_final.columns:
            cols.append(col_name)

# 4️⃣ Todas as std de tempo
for suffix in ['time_std']:
    for dist in distances:
        col_name = f'{dist}_{suffix}'
        if col_name in df_final.columns:
            cols.append(col_name)

df_final = df_final[cols].round(4)

# Ordena por dataset
df_final = df_final.sort_values('dataset').reset_index(drop=True)

# Salva CSV final
df_final.to_csv('./agregacaoo/summary_results.csv', index=False)

print("CSV final gerado: summary_results.csv")
