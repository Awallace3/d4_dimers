import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
import qcelemental as qcel
from qm_tools_aw import tools
import os
import apnet_pt


def compute_apnet_energies(mols):
    import apnet_pt

    interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
        mols,
        compile=False,
        batch_size=2,
    )
    return interaction_energies


def ap3_d_elst_classical_energies(mols):
    path_to_qcml = os.path.join(os.path.expanduser("~"), "gits/qcmlforge/models")
    am_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_3.pt"
    at_hf_vw_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_h+1_3.pt"
    at_elst_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_elst_h+1_3.pt"
    ap3_path = f"{path_to_qcml}/../models/ap3_ensemble/3/ap3_.pt"
    atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        use_GPU=False,
        n_neuron=64,
        n_params=1,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        model_type="AtomTypeParamNN",
        # model_type="AtomTypeParamMPNN",
        # pre_trained_model_path=at_elst_path_mpnn,
        pre_trained_model_path=at_elst_path,
    )
    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        pre_trained_model_path=ap3_path,
    )
    pred, pair_elst, pair_ind = ap3.predict_qcel_mols(
        mols, batch_size=16, return_classical_pairs=True
    )
    return pred


def create_df_l14_data():
    df = pd.read_csv("./l14_molecules.csv")
    df = df.rename({"Molecule": "L14"}, axis=1)
    df2 = pd.read_csv("./l14_data.csv")
    df = pd.merge(df, df2, on="L14", how="inner")
    df["qcel_mol"] = df["XYZ"].apply(tools.xyz_dimer_to_qcelemental_mol_dimer)
    # Print which rows have missing dimer geometries to be dropped
    print("Rows with missing dimer geometries (to be dropped):")
    print(df[df["qcel_mol"].isna()][["L14"]])
    print("Remaining rows after dropping missing geometries:")
    print(df[df["qcel_mol"].notna()][["L14"]])
    df = df.drop(columns=["XYZ"])
    df = df.dropna(subset=["qcel_mol"])
    ies = compute_apnet_energies(df["qcel_mol"].to_list())
    df["AP2 total"] = ies[:, 0]
    df["AP2 elst"] = ies[:, 1]
    df["AP2 exch"] = ies[:, 2]
    df["AP2 indu"] = ies[:, 3]
    df["AP2 disp"] = ies[:, 4]
    df["AP2 elst+exch+indu"] = ies[:, 1:4].sum(axis=1)
    ies = ap3_d_elst_classical_energies(df["qcel_mol"].to_list())
    df["AP3 total"] = np.sum(ies[:, 0:4], axis=1)
    df["AP3 elst"] = ies[:, 0]
    df["AP3 exch"] = ies[:, 1]
    df["AP3 indu"] = ies[:, 2]
    df["AP3 disp"] = ies[:, 3]
    print(df)
    df.to_pickle("./l14_data_with_apnet.pkl")
    return


def visualize_errors(
    data_path: str = "./l14_data.csv",
    out_png: str = "l14_violin_plots.png",
    reference_col="canonical_CCSD(T)",
):
    """Create violin plots of method errors vs canonical_CCSD(T) with MAE annotations.

    - Reads `data_path` into a DataFrame
    - Computes error = method - canonical_CCSD(T) for each method
    - Plots violin plots of error distributions, sorted by MAE
    - Annotates each violin with its MAE (kcal/mol)
    """

    if data_path.endswith(".pkl"):
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_csv(data_path)

    print(df[["L14", reference_col, "MP2"]])
    print(df[["L14", reference_col, "SCS(MI)-MP2"]])
    print(df[["L14", reference_col, 'SAPT0', 'AP3 total', 'AP3 elst', 'AP3 exch', 'AP3 indu', 'AP3 disp']])
    print(df[["L14", reference_col, 'SAPT0', 'AP2 total', 'AP2 elst', 'AP2 exch', 'AP2 indu', 'AP2 disp']])
    print(df[["L14", reference_col, "SCS(MI)-MP2"]])
    print(df.columns.to_list())
    # Columns to exclude from the set of energy/method columns
    exclude_cols = {
        "L14",
        reference_col,
        "Type",
        "(T)_contribution",
        # "canonical_CCSD(cT)-fit/CBS",
        "delta_E",
        "Eelst",
        "Eexch",
        "AP2 exch",
        "AP2 indu",
        "AP2 disp",
        "Eind",
        "Edisp",
        "Etotal",
        "Edisp/Eelst",
    }

    # Candidate method columns: everything except excluded items
    method_cols = [c for c in df.columns if c not in exclude_cols]

    # Keep only numeric columns (pandas may parse some columns as object)
    numeric_methods = []
    for col in method_cols:
        # Try to coerce to numeric and check how many non-nan values we get
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() >= max(3, int(0.25 * len(df))):
            numeric_methods.append(col)

    errors_dict = {}
    maes = {}

    for col in numeric_methods:
        mask = (
            df[reference_col].notna() & pd.to_numeric(df[col], errors="coerce").notna()
        )
        if mask.sum() == 0:
            continue
        errors = (
            pd.to_numeric(df[col], errors="coerce")[mask]
            - pd.to_numeric(df[reference_col], errors="coerce")[mask]
        )
        # print(col, errors)
        if errors.size == 0:
            continue
        errors_dict[col] = errors.values
        # maes[col] = np.mean(np.abs(errors.values))
        print(col, len(errors.values), np.mean(np.abs(errors.values)))
        maes[col] = np.mean(np.abs(errors.values))

    if not maes:
        raise RuntimeError("No numeric method columns found to plot.")

    # Sort methods by MAE (ascending)
    sorted_methods = sorted(maes.keys(), key=lambda k: maes[k])

    data_to_plot = [errors_dict[m] for m in sorted_methods]

    # Figure size scales with number of methods
    n_methods = len(sorted_methods)
    width = max(10, n_methods * 0.4)
    fig, ax = plt.subplots(figsize=(width, 6))

    positions = np.arange(1, n_methods + 1)

    parts = ax.violinplot(
        data_to_plot, positions=positions, showmeans=True, showmedians=True
    )

    # Style the violins (optional but improves readability)
    for pc in parts.get("bodies", []):
        pc.set_facecolor("#D0E1F9")
        pc.set_edgecolor("black")
        pc.set_alpha(0.9)
    if "cmeans" in parts:
        parts["cmeans"].set_color("red")
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")

    # Horizontal line at zero error
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    # Determine y-limits with a margin so annotations fit
    all_errors = np.concatenate([d for d in data_to_plot if len(d) > 0])
    err_min, err_max = np.nanmin(all_errors), np.nanmax(all_errors)
    y_margin = max(0.5, 0.05 * max(abs(err_min), abs(err_max)))
    ax.set_ylim(err_min - y_margin, err_max + y_margin * 3)

    y_top = ax.get_ylim()[1]

    # Annotate MAE above each violin
    for i, method in enumerate(sorted_methods):
        mae = maes[method]
        x = positions[i]
        # slightly below top to avoid clipping
        y = y_top - 0.02 * (y_top - ax.get_ylim()[0])
        ax.text(
            x,
            y,
            f"MAE\n{mae:.2f}",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        sorted_methods, rotation=90 if n_methods > 10 else 45, ha="right", fontsize=8
    )

    ax.set_ylabel("Error vs canonical CCSD(T) (kcal/mol)")
    ax.set_title("Error Distributions vs canonical_CCSD(T) — Violins with MAE")

    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved violin plot to {out_png}")

    # also print MAE summary
    print("\nMAE summary (kcal/mol):")
    for m in sorted_methods:
        print(f"{m:30s}: {maes[m]:6.3f}")

    # plt.show()


def main():
    # create_df_l14_data()
    # return
    visualize_errors(data_path="./l14_data_with_apnet.pkl")
    visualize_errors(
        data_path="./l14_data_with_apnet.pkl",
        out_png="l14_dft_violin_plots_cbs.png",
        reference_col="canonical_CCSD(cT)-fit/CBS",
    )


if __name__ == "__main__":
    main()
