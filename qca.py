import pandas as pd
import numpy as np
from pprint import pprint as pp
import qm_tools_aw
from qm_tools_aw import qca
import qcelemental as qcel
from qcportal import PortalClient
from pprint import pprint as pp
from qcportal.singlepoint import (
    SinglepointDataset,
    SinglepointDatasetEntry,
    QCSpecification,
)

client = qm_tools_aw.qca.establish_client("http://localhost:7777", verify=False)
# client = PortalClient("http://localhost:7777", verify=False)
print(client)


def setup_entries():
    df = pd.read_pickle("./plots/ddft_study.pkl")
    df["Geometry"] = df["Geometry"].apply(lambda x: np.array(x))
    df["monAs"] = df["monAs"].apply(lambda x: np.array(x))
    df["monBs"] = df["monBs"].apply(lambda x: np.array(x))
    # pp(df.columns.tolist())
    # return

    def create_qcel_mol(r):
        ma, mb = r["monAs"], r["monBs"]
        g1, g2 = r["Geometry"][ma], r["Geometry"][mb]
        m1 = f"{r['charges'][1][0]} {r['charges'][1][1]}\n"
        m1 += qm_tools_aw.tools.print_cartesians_pos_carts(
            g1[:, 0], g1[:, 1:], only_results=True
        )
        m2 = f"{r['charges'][2][0]} {r['charges'][2][1]}\n"
        m2 += qm_tools_aw.tools.print_cartesians_pos_carts(
            g2[:, 0], g2[:, 1:], only_results=True
        )
        mol = qcel.models.Molecule.from_data(
            m1 + "--\n" + m2,
            extras={
                "id": "id",
                "system_id": str(r["system_id"]),
                "DB": str(r["DB"]),
                "R": str(r["R"]),
                "Benchmark": float(r["Benchmark"]),
                # PBE0
                "SAPT_DFT_pbe0_adz_elst": float(r["SAPT_DFT_pbe0_adz_elst"]),
                "SAPT_DFT_pbe0_adz_exch": float(r["SAPT_DFT_pbe0_adz_exch"]),
                "SAPT_DFT_pbe0_adz_indu": float(r["SAPT_DFT_pbe0_adz_indu"]),
                "SAPT_DFT_pbe0_adz_disp": float(r["SAPT_DFT_pbe0_adz_disp"]),
                "pbe0_adz_grac_A": float(r["pbe0_adz_grac_A"]),
                "pbe0_adz_grac_B": float(r["pbe0_adz_grac_B"]),
                "SAPT_DFT_pbe0_adz_dHF": float(r["SAPT_DFT_pbe0_adz_dHF"]),
                "SAPT_DFT_pbe0_adz_dDFT": float(r["SAPT_DFT_pbe0_adz_dDFT"]),
                "SAPT_DFT_pbe0_adz_D4_IE": float(r["SAPT_DFT_pbe0_adz_D4_IE"]),
                "SAPT_DFT_pbe0_adz_DFT_IE": float(r["SAPT_DFT_pbe0_adz_DFT_IE"]),
                "pbe0_atz_grac_A": float(r["pbe0_atz_grac_A"]),
                "pbe0_atz_grac_B": float(r["pbe0_atz_grac_B"]),
                # B3LYP
                "SAPT_DFT_pbe0_atz_elst": float(r["SAPT_DFT_pbe0_atz_elst"]),
                "SAPT_DFT_pbe0_atz_exch": float(r["SAPT_DFT_pbe0_atz_exch"]),
                "SAPT_DFT_pbe0_atz_indu": float(r["SAPT_DFT_pbe0_atz_indu"]),
                "SAPT_DFT_pbe0_atz_disp": float(r["SAPT_DFT_pbe0_atz_disp"]),
                "SAPT_DFT_pbe0_atz_dHF": float(r["SAPT_DFT_pbe0_atz_dHF"]),
                "SAPT_DFT_pbe0_atz_dDFT": float(r["SAPT_DFT_pbe0_atz_dDFT"]),
                "SAPT_DFT_pbe0_atz_D4_IE": float(r["SAPT_DFT_pbe0_atz_D4_IE"]),
                "SAPT_DFT_pbe0_atz_DFT_IE": float(r["SAPT_DFT_pbe0_atz_DFT_IE"]),
                "b3lyp_adz_grac_A": float(r["b3lyp_adz_grac_A"]),
                "b3lyp_adz_grac_B": float(r["b3lyp_adz_grac_B"]),
                "SAPT_DFT_b3lyp_adz_elst": float(r["SAPT_DFT_b3lyp_adz_elst"]),
                "SAPT_DFT_b3lyp_adz_exch": float(r["SAPT_DFT_b3lyp_adz_exch"]),
                "SAPT_DFT_b3lyp_adz_indu": float(r["SAPT_DFT_b3lyp_adz_indu"]),
                "SAPT_DFT_b3lyp_adz_disp": float(r["SAPT_DFT_b3lyp_adz_disp"]),
                "SAPT_DFT_b3lyp_adz_dHF": float(r["SAPT_DFT_b3lyp_adz_dHF"]),
                "SAPT_DFT_b3lyp_adz_dDFT": float(r["SAPT_DFT_b3lyp_adz_dDFT"]),
                "SAPT_DFT_b3lyp_adz_D4_IE": float(r["SAPT_DFT_b3lyp_adz_D4_IE"]),
                "SAPT_DFT_b3lyp_adz_DFT_IE": float(r["SAPT_DFT_b3lyp_adz_DFT_IE"]),
                "SAPT_DFT_b3lyp_atz_elst": float(r["SAPT_DFT_b3lyp_atz_elst"]),
                "SAPT_DFT_b3lyp_atz_exch": float(r["SAPT_DFT_b3lyp_atz_exch"]),
                "SAPT_DFT_b3lyp_atz_indu": float(r["SAPT_DFT_b3lyp_atz_indu"]),
                "SAPT_DFT_b3lyp_atz_disp": float(r["SAPT_DFT_b3lyp_atz_disp"]),
                "SAPT_DFT_b3lyp_atz_dHF": float(r["SAPT_DFT_b3lyp_atz_dHF"]),
                "SAPT_DFT_b3lyp_atz_dDFT": float(r["SAPT_DFT_b3lyp_atz_dDFT"]),
                "SAPT_DFT_b3lyp_atz_D4_IE": float(r["SAPT_DFT_b3lyp_atz_D4_IE"]),
                "SAPT_DFT_b3lyp_atz_DFT_IE": float(r["SAPT_DFT_b3lyp_atz_DFT_IE"]),
            },
        )
        return mol

    df["qcel_mol"] = df.apply(lambda r: create_qcel_mol(r), axis=1)
    print(df[["system_id", "qcel_mol"]])
    print(df["qcel_mol"].iloc[0])
    geoms = []
    for i, row in df.iterrows():
        geoms.append([row["system_id"], row["qcel_mol"]])
    ds = qca.init_singlepoint_dataset(
        client,
        ds_name="ddft_study",
        geoms=geoms,
    )
    return ds


def main():
    ds = client.get_dataset("singlepoint", "ddft_study")
    client.delete_dataset(ds.id, delete_records=True)
    setup_entries()
    # return
    ds = qca.init_singlepoint_dataset(
        client,
        ds_name="ddft_study",
    )
    print(ds)
    functional = "pbe0"
    # return
    qca.create_singlepoint_dataset_specification(
        ds,
        program="psi4",
        driver="energy",
        method="sapt(dft)",
        basis="aug-cc-pvtz",
        keywords={
            "maxiter": 250,
            "E_CONVERGENCE": 8,
            "D_CONVERGENCE": 8,
            "freeze_core": "True",
            "guess": "sad",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": functional,
            "SAPT_DFT_DO_DISP": True,
            "SAPT_DFT_DO_DDFT": True,
            "SAPT_DFT_D4_IE": True,
            "SAPT_DFT_GRAC_COMPUTE": "ITERATIVE",
        },
        protocols={"stdout": True},
        specification_name=f"psi4/sapt({functional})/aug-cc-pvdz",
        compute_tag="hive",
    )
    print(ds.status())
    return


if __name__ == "__main__":
    main()
