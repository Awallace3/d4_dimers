import numpy as np
import pandas as pd
from qm_tools_aw import tools
import subprocess

def compute_dispml_row(row: pd.Series, dispml_model="D3-ML", path_dispml="./dispml/"):
    row['monAs'] = np.array(row['monAs'])
    row['monBs'] = np.array(row['monBs'])
    row['Geometry'] = np.array(row['Geometry'])
    geom_A = row['Geometry'][row['monAs']]
    geom_B = row['Geometry'][row['monBs']]
    tools.write_cartesians_to_xyz(geom_A[:, 0], geom_A[:, 1:], "monA.xyz")
    tools.write_cartesians_to_xyz(geom_B[:, 0], geom_B[:, 1:], "monB.xyz")
    dispml_output = subprocess.check_output(
        f"python3 {path_dispml}main.py --model {dispml_model} monA.xyz monB.xyz",
        shell=True,
    )
    ml_disp = float(dispml_output.decode("utf-8").split(":")[-1].split()[-1])
    print(ml_disp)
    return ml_disp

def compute_dispml_df(df: pd.DataFrame, dispml_model="D3-ML", path_dispml="./dispml/"):
    df[dispml_model] = df.apply(lambda row: compute_dispml_row(row, dispml_model, path_dispml), axis=1)
    return df
