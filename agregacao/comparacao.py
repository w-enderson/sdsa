import pandas as pd
import os


def combine_statistics_results(directory):
    # Lista de todos os arquivos que começam com 'statistics_results'
    statistics_files = [f for f in os.listdir(directory) if f.startswith('statistics_results')]

    # DataFrame vazio para acumular os resultados
    combined_df = pd.DataFrame()

    # Iterar sobre os arquivos e processá-los
    for file in statistics_files:
        # Obter o caminho completo do arquivo
        file_path = os.path.join(directory, file)

        # Carregar o arquivo CSV
        df = pd.read_csv(file_path)

        # Extrair o nome da distância (como "City_Block", "Euclidean_distance", etc.)
        distance_name = file.split('_')[-1].replace('.csv', '')  # Parte após "statistics_results"

        # Renomear as colunas para incluir o nome da distância
        df = df.rename(columns={
            'acc_mean': f'acc_mean_{distance_name}',
            'acc_std': f'acc_std_{distance_name}',
            'time_mean': f'time_mean_{distance_name}',
            'time_std': f'time_std_{distance_name}'
        })

        # Adicionar o DataFrame processado ao DataFrame combinado
        if combined_df.empty:
            # Para o primeiro arquivo, apenas adicionar
            combined_df = df
        else:
            # Para os outros arquivos, fazer o merge com base nas colunas "dataset" e "classifier"
            combined_df = pd.merge(combined_df, df, on=['dataset', 'classifier'], how='outer')

    # Salvar o DataFrame combinado em um arquivo CSV
    combined_df.to_csv(os.path.join(directory, 'combined_statistics_results.csv'), index=False)

    return combined_df


# Chamada da função, fornecendo o diretório onde estão os arquivos CSV
directory = './'
combined_df = combine_statistics_results(directory)

# Exibir os primeiros registros do DataFrame combinado
