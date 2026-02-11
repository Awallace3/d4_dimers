import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
import xyz2mol

def compute(df):
    from src import dispml_calls
    df = df.dropna(subset=['benchmark ref energy'])
    df = dispml_calls.compute_dispml_df(df, print_updates=True)
    print(df['D3-ML'])
    # df.to_pickle("./plots/ddft_study.pkl")
    return

def main():
    df = pd.read_pickle("./plots/ddft_study.pkl")
    print(df)
    # compute how many null D3-ML
    print(df['D3-ML'].isnull().sum())
    compute(df)
    return


if __name__ == "__main__":
    main()
