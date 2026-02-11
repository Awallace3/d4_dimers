import numpy as np
import pandas as pd
from qm_tools_aw import tools
import subprocess
from qcelemental import constants
from tqdm import tqdm

bohr_to_angstrom = constants.conversion_factor("bohr", "angstrom")

def compute_dispml_row(row: pd.Series, dispml_model="D3-ML", path_dispml="./dispml/", bohr2ang=False, print_updates=False):
    row['monAs'] = np.array(row['monAs'])
    row['monBs'] = np.array(row['monBs'])
    row['Geometry'] = np.array(row['Geometry'])
    if row['monAs'].ndim == 0:
        row['monAs'] = np.array([row['monAs']])
    if row['monBs'].ndim == 0:
        row['monBs'] = np.array([row['monBs']])
    # print(row['Geometry'])
    # print(row['monAs'])
    # print(row['monBs'])
    # print(row['charges'])
    # try:
    geom_A = row['Geometry'][row['monAs']]
    geom_B = row['Geometry'][row['monBs']]
    # ensure that geom_A and geom_B are 2D arrays
    if geom_A.ndim == 1:
        geom_A = geom_A.reshape(1, -1)
    if geom_B.ndim == 1:
        geom_B = geom_B.reshape(1, -1)
    if bohr2ang:
        geom_A[:, 1:] *= bohr_to_angstrom
        geom_B[:, 1:] *= bohr_to_angstrom
    tools.write_cartesians_to_xyz(geom_A[:, 0], geom_A[:, 1:], "monA.xyz", charge_multiplicity=row['charges'][1], charge=True, multiplicty=False)
    tools.write_cartesians_to_xyz(geom_B[:, 0], geom_B[:, 1:], "monB.xyz", charge_multiplicity=row['charges'][2], charge=True, multiplicty=False)
    dispml_output = subprocess.check_output(
        f"python3 {path_dispml}main.py --model {dispml_model} monA.xyz monB.xyz",
        shell=True,
    ).decode("utf-8")
    # If python call fails, capture the error and print it
    if dispml_output is None:
        print(f"Error in dispml for {row['id'] = } from {row['DB']} {row['system_id']}: No output from dispml")
        return np.nan
    print(dispml_output)
    if "Valence of atom" in dispml_output:
        print(f"Error in dispml for {row['id'] = } from {row['DB']} {row['system_id']}: Valence of atom error")
        return np.nan
    ml_disp = float(dispml_output.split(":")[-1].split()[-1])
    if print_updates:
        print(f"{row['id'] = } {ml_disp = :.2f}, {row['-D4 (SAPT0_adz_3_IE)'] = :.2f}")
    # except (Exception) as e:
    #     if print_updates:
    #         print(f"Error in dispml for {row['id'] = } from {row['DB']}:", e)
    #         print(f"{row['system_id'] = }")
    #         print(dispml_output)
    #         tools.print_cartesians(row['Geometry'])
    #     ml_disp = np.nan
    return ml_disp

def compute_dispml_df(df: pd.DataFrame, dispml_model="D3-ML", path_dispml="./dispml/", print_updates=False):
    # Select rows where the dispml_model column is NaN
    nan_rows = df[df[dispml_model].isna()]
    
    # Iterate only over rows with NaN in `dispml_model`
    for i, row in tqdm(nan_rows.iterrows(), total=len(nan_rows)):
        # `i` is the index in the df where the `dispml_model` is NaN
        df.at[i, dispml_model] = compute_dispml_row(row, dispml_model, path_dispml,
                                                    print_updates=print_updates)
        
    return df

def ensure_geometry_angstrom(r):
    r['Geometry'] = np.array(r['Geometry'])
    distances = np.linalg.norm(r['Geometry'][:, 1:] - r['Geometry'][:, 1:].reshape(-1, 1, 3), axis=-1)
    np.fill_diagonal(distances, 10)
    n = r['id']
    if np.all(distances > 1.5):
        print(f"Geometry {n} has all distances larger than 1.5, {r['DB']} {r['system_id']}")
        tools.print_cartesians(r['Geometry'])
        r['Geometry'][:, 1:] *= bohr_to_angstrom
        print("\nConverted to Angstrom\n")
        tools.print_cartesians(r['Geometry'])
    return r


def check_geometry_units(df, cutoff=1.5):
    df = df.apply(ensure_geometry_angstrom, axis=1)
    # for n, r in df.iterrows():
    #     # check if all distances are greater than 1.5 Angstrom
    #     r['Geometry'] = np.array(r['Geometry'])
    #     distances = np.linalg.norm(r['Geometry'][:, 1:] - r['Geometry'][:, 1:].reshape(-1, 1, 3), axis=-1)
    #     # if np.any(distances < 1.5):
    #     #     print(f"Geometry {n} has distances smaller than 1.5 Angstrom")
    #     # make diagonal 10 to avoid self-interaction
    #     np.fill_diagonal(distances, 10)
    #     if np.all(distances > cutoff):
    #         print(f"Geometry {n} has all distances larger than {cutoff} Angstrom")
    #         tools.print_cartesians(r['Geometry'])
    #         r['Geometry'][:, 1:] *= bohr_to_angstrom
    #         df.loc[n, 'Geometry'] = list(r['Geometry'])
    return df

def main():
    check_geometry_units(df)
    return


if __name__ == "__main__":
    main()
