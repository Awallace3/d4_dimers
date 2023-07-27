import numpy as np
import pandas as pd
import src
import qcelemental as qcel
import tqdm
from qm_tools_aw import tools
from pandarallel import pandarallel

pandarallel.initialize(use_memory_fs=True)

df = pd.read_pickle("data/grimme_fitset_test2.pkl")
df = df[
    [
        "HF_qz",
        "DB",
        "Geometry_bohr",
        "Benchmark",
        "charges",
        "monAs",
        "monBs",
        "C6s",
        "C6_A",
        "C6_B",
        "C6_ATM",
        "C6_ATM_A",
        "C6_ATM_B",
    ]
].copy()
print(df)


def multiply(x):
    x["HF_qz"]
    return x["HF_qz"] * 0.5


df["test"] = df.parallel_apply(multiply, axis=1)
print(df)
df["test2"] = df.parallel_apply(
    lambda r: src.locald4.compute_bj_dimer_f90([1, 1, 1], r), axis=1
)
print(df)
