import pandas as pd

df1= pd.read_csv("synthetic_datasets/synthetic-dataset4.csv ")

df1.shape

df1["target"].value_counts()
