import pandas as pd

def load_data():
    df = pd.read_csv("data/raw/creditcard.csv")
    print(df.head())
