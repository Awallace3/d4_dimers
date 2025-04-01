from qm_tools_aw import tools
from glob import glob
import numpy as np
import pandas as pd
import hrcl_jobs
import hrcl_jobs_psi4

print(hrcl_jobs.import_error)


def read_xyzs(
    geom_dir="water",
    monA_len=3,
    monB_len=3,
    grac_shift=0.1307,
):
    files = glob(f"{geom_dir}/*.xyz")
    data = {
        "system_id": [],
        "Geometry": [],
    }
    for file in files:
        distance = file.split("/")[-1].split("_")[-1].split(".xyz")[0]
        if distance == "animation":
            continue
        pos, carts = tools.read_xyz_to_pos_carts(file)
        pos = np.reshape(pos, (-1, 1))
        geom = np.concatenate((pos, carts), axis=1)
        data["system_id"].append(f"{geom_dir}_{distance}")
        data["Geometry"].append(geom)
    data["charges"] = [
        np.array([[0, 1] for i in range(3)]) for i in range(len(data["Geometry"]))
    ]
    data["monAs"] = [np.array(range(monA_len))
                     for i in range(len(data["Geometry"]))]
    data["monBs"] = [
        np.array(range(monA_len, monA_len + monB_len))
        for i in range(len(data["Geometry"]))
    ]
    data['pbe0_grac_shift_a'] = [grac_shift for i in range(len(data["Geometry"]))]
    data['pbe0_grac_shift_b'] = [grac_shift for i in range(len(data["Geometry"]))]
    df = pd.DataFrame(data)
    return df


def main():
    df_water = read_xyzs("water", 3, 3, 0.1307)
    # benzene ionization potential = 0.072113
    df_benzene = read_xyzs("benzene", 12, 12, 0.072113)
    df = pd.concat([df_water, df_benzene], ignore_index=True)
    df['id'] = df.index
    df.reset_index(drop=True, inplace=True)
    print(df.columns.values.tolist())
    print(df[['Geometry']])
    df.to_pickle("curves.pkl")
    DB_NAME, TABLE_NAME = "curves.db", "main"
    hrcl_jobs.sqlt.convert_df_into_sql(
        "curves.pkl",
        DB_NAME,
        table_name=TABLE_NAME,
        input_columns={
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "system_id": "TEXT",
            "Geometry": "array",
            "charges": "array",
            "monAs": "array",
            "monBs": "array",
            "pbe0_grac_shift_a": "REAL",
            "pbe0_grac_shift_b": "REAL",
        },
        output_columns={},
        overwrite=True,
    )
    hrcl_jobs.dataset.compute_energy(
        DB_NAME,
        TABLE_NAME,
        col_check="SAPT_DFT_pbe0_adz",
        options={
            "maxiter": 250,
            "E_CONVERGENCE": 8,
            "D_CONVERGENCE": 8,
            "freeze_core": "True",
            "guess": "sad",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "pbe0",
            "SAPT_DFT_DO_DDFT": True,
            "SAPT_DFT_D4_IE": True,
        }
    )
    return


if __name__ == "__main__":
    main()
