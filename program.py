# program.py
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Executa scripts Python com um classificador.")
    parser.add_argument("-n", "--name", required=True, help="Nome do classificador")
    args = parser.parse_args()

    classifier = args.name

    # Lista de comandos que queremos rodar
    commands = [
        f"python main.py -c {classifier}_not_update",
        f"python main.py -c {classifier}",
        f"python main_synthetic_datasets.py -c {classifier}",
        f"python main_synthetic_datasets.py -c {classifier}_not_update"
    ]

    # Executar cada comando sequencialmente
    for cmd in commands:
        print(f"Executando: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Erro ao executar: {cmd}")
            break  # opcional: parar se algum comando falhar

if __name__ == "__main__":
    main()
