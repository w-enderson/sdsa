import pandas as pd


for i in range(1,5):

    df1= pd.read_csv(f"synthetic_datasets/synthetic-dataset{i}.csv")

    print(f"\n Dataset {i}")
    print(df1.shape)

    print(df1["target"].value_counts())

