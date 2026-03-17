from qm_tools_aw import tools
from glob import glob
import numpy as np
import pandas as pd
import qcelemental as qcel
from pprint import pprint as pp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def compute_N(df_l, col_E, sign_flip=True, print_lvl=0, min_distance=None):
    """
    Compute the power law exponent N for the relationship f(R) = C/R^N.

    Parameters:
    -----------
    df_l : DataFrame
        DataFrame containing distance and energy data
    col_E : str
        Column name for the energy values
    sign_flip : bool
        Whether to flip the sign of the energy values (for attractive interactions)
    print_lvl : int
        Level of verbosity for printing
    min_distance : float or None
        Minimum distance to include in the fit, helps exclude short-range data

    Returns:
    --------
    N : float
        The power law exponent
    """
    # Filter for negative energies (attractive interactions)
    df_l_neg = df_l[df_l[col_E] < 0].copy()

    # Apply minimum distance filter if provided
    if min_distance is not None:
        df_l_neg = df_l_neg[df_l_neg["R"] >= min_distance]

    R = df_l_neg["R"].values
    f_R = df_l_neg[col_E].values

    if sign_flip:
        f_R = -f_R

    # Define the power law function for direct fitting
    def power_law(r, c, n):
        return c / (r**n)

    # Use curve_fit for direct fitting without logarithm transformation
    try:
        # Initial guess: C=1, N=6 (reasonable for dispersion)
        popt, pcov = curve_fit(power_law, R, f_R, p0=[1.0, 6.0])
        C, N = popt

        # Calculate R-squared to assess fit quality
        residuals = f_R - power_law(R, C, N)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((f_R - np.mean(f_R)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        if print_lvl > 0:
            print(
                f"{col_E} N: {N:.2f}, C: {C:.4e}, R²: {r_squared:.4f} on {len(df_l_neg)} points"
            )

        return N

    except RuntimeError:
        # Fall back to log-linear fit if curve_fit fails
        log_R = np.log(R)
        log_f_R = np.log(f_R)
        slope, intercept = np.polyfit(log_R, log_f_R, 1)
        N = -slope
        C = np.exp(intercept)

        if print_lvl > 0:
            print(f"{col_E} N: {N:.2f} (log-linear fallback) on {len(df_l_neg)} points")

        return N


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
    data["monAs"] = [np.array(range(monA_len)) for i in range(len(data["Geometry"]))]
    data["monBs"] = [
        np.array(range(monA_len, monA_len + monB_len))
        for i in range(len(data["Geometry"]))
    ]
    data["pbe0_grac_shift_a"] = [grac_shift for i in range(len(data["Geometry"]))]
    data["pbe0_grac_shift_b"] = [grac_shift for i in range(len(data["Geometry"]))]
    df = pd.DataFrame(data)
    return df


def full():
    import hrcl_jobs

    df_water = read_xyzs("water", 3, 3, 0.1307)
    # benzene ionization potential = 0.072113
    df_benzene = read_xyzs("benzene", 12, 12, 0.072113)
    df = pd.concat([df_water, df_benzene], ignore_index=True)
    df["id"] = df.index
    df.reset_index(drop=True, inplace=True)
    print(df.columns.values.tolist())
    print(df[["Geometry"]])
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
        overwrite=False,
    )
    hrcl_jobs.dataset.compute_energy(
        DB_NAME,
        TABLE_NAME,
        col_check="SAPT_DFT_pbe0_atz",
        options={
            "maxiter": 250,
            "E_CONVERGENCE": 8,
            "D_CONVERGENCE": 8,
            "freeze_core": "True",
            "guess": "sad",
            "scf_type": "df",
            "SAPT_DFT_FUNCTIONAL": "pbe0",
            "SAPT_DFT_DO_DDFT": False,
            "SAPT_DFT_D4_IE": False,
        },
        output_root="curve_outputs",
        hive_params={
            "mem_per_process": "80 gb",
            "num_omp_threads": 16,
        },
    )
    hrcl_jobs.dataset.compute_energy(
        DB_NAME,
        TABLE_NAME,
        col_check="SAPT0_adz",
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
        },
        output_root="curve_outputs",
        hive_params={
            "mem_per_process": "80 gb",
            "num_omp_threads": 16,
        },
    )
    # hrcl_jobs.sqlt.table_to_df_pkl(
    #     db_p=DB_NAME,
    #     table=TABLE_NAME,
    #     df_p="curves.pkl",
    # )
    return


def plot_results(fit=True):
    h2kcal = qcel.constants.conversion_factor("hartree", "kcal/mol")
    df = pd.read_pickle("./curves_d4.pkl")
    print(df)
    pp(df.columns.tolist())
    df["R"] = [float(i.split("_")[-1]) for i in df["system_id"]]
    df = df[df["R"] < 8].copy()
    df["system_type"] = [i.split("_")[0] for i in df["system_id"]]
    print(df)
    pp(df.columns.tolist())
    df["SAPT_DFT_pbe_atz_total"] = [np.nan for i in range(len(df))]
    df["SAPT_DFT_pbe_atz_elst"] = [np.nan for i in range(len(df))]
    df["SAPT_DFT_pbe_atz_exch"] = [np.nan for i in range(len(df))]
    df["SAPT_DFT_pbe_atz_indu"] = [np.nan for i in range(len(df))]
    df["SAPT_DFT_pbe_atz_disp"] = [np.nan for i in range(len(df))]
    df["SAPT_DFT_pbe_adz_total"] = [np.nan for i in range(len(df))]
    df["SAPT_DFT_pbe_adz_elst"] = [np.nan for i in range(len(df))]
    df["SAPT_DFT_pbe_adz_exch"] = [np.nan for i in range(len(df))]
    df["SAPT_DFT_pbe_adz_indu"] = [np.nan for i in range(len(df))]
    df["SAPT_DFT_pbe_adz_disp"] = [np.nan for i in range(len(df))]
    df["pbe_d4_sapt_adz_disp"] = [np.nan for i in range(len(df))]
    # locald4
    vars_json_files = glob("./curve_outputs/*/*/*.json")
    for i in vars_json_files:
        print(i)
        v = tools.json_to_dict(i)
        if "pvdz" in i:
            id = int(i.split("/")[2])
            df.loc[id, "SAPT_DFT_pbe_adz_total"] = v["SAPT TOTAL ENERGY"] * h2kcal
            df.loc[id, "SAPT_DFT_pbe_adz_elst"] = v["SAPT ELST ENERGY"] * h2kcal
            df.loc[id, "SAPT_DFT_pbe_adz_exch"] = v["SAPT EXCH ENERGY"] * h2kcal
            df.loc[id, "SAPT_DFT_pbe_adz_indu"] = v["SAPT IND ENERGY"] * h2kcal
            df.loc[id, "SAPT_DFT_pbe_adz_disp"] = v["SAPT DISP ENERGY"] * h2kcal
            df.loc[id, "pbe_d4_sapt_adz_disp"] = (
                v["D4 IE"] + v["SAPT(DFT) DELTA DFT"] - v["SAPT(DFT) DELTA HF"]
            ) * h2kcal
        else:
            id = int(i.split("/")[2])
            df.loc[id, "SAPT_DFT_pbe_atz_total"] = v["SAPT TOTAL ENERGY"] * h2kcal
            df.loc[id, "SAPT_DFT_pbe_atz_elst"] = v["SAPT ELST ENERGY"] * h2kcal
            df.loc[id, "SAPT_DFT_pbe_atz_exch"] = v["SAPT EXCH ENERGY"] * h2kcal
            df.loc[id, "SAPT_DFT_pbe_atz_indu"] = v["SAPT IND ENERGY"] * h2kcal
            df.loc[id, "SAPT_DFT_pbe_atz_disp"] = v["SAPT DISP ENERGY"] * h2kcal
    print(df)
    # Plot SAPT(DFT) disp and pbe-d4 disp, have subplots 2x1 (water, benzene), fit the N for > 3 Angstroms (R > 3)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6, 6),
        # sharex=True,
    )

    systems = ["water", "benzene"]
    system_starting_distance = [3.21, 4.56]
    colors = {
        "SAPT_DFT_pbe_atz_disp": "blue",
        "pbe_d4_sapt_adz_disp": "red",
        "SAPT(PBE0)-D4/aDZ (I)": "green",
    }
    markers = {
        "SAPT_DFT_pbe_atz_disp": "o",
        "pbe_d4_sapt_adz_disp": "s",
        "SAPT(PBE0)-D4/aDZ (I)": "^",
    }
    labels = {
        "SAPT_DFT_pbe_atz_disp": "SAPT(PBE0)/aTZ",
        "pbe_d4_sapt_adz_disp": "PBE0-D4/aDZ",
        "SAPT(PBE0)-D4/aDZ (I)": "SAPT(PBE0)-D4/aDZ (I)",
    }

    for i, system in enumerate(systems):
        df_system = df[df["system_type"] == system].sort_values("R")
        print(df_system["R"])

        if fit:
            axes[i].axvline(
                system_starting_distance[i],
                color="black",
                linestyle="--",
                label="Fit Start",
            )
        # get index of minimum total energy of df_system and plot vertical grey line at that distance
        min_index = df_system["SAPT_DFT_pbe_atz_total"].idxmin()
        min_distance = df_system.loc[min_index, "R"]
        print(f"Minimum distance for {system}: {min_distance}")
        axes[i].axvline(
            min_distance, color="grey", linestyle="--", label="Equilibrium Distance"
        )
        for col, color in colors.items():
            # Plot actual data points
            if fit:
                axes[i].scatter(
                    df_system["R"],
                    df_system[col],
                    color=color,
                    marker=markers[col],
                    label=f"{labels[col]} (Data)",
                )
            else:
                axes[i].plot(
                    df_system["R"],
                    df_system[col],
                    color=color,
                    marker=markers[col],
                    label=f"{labels[col]}",
                )
            axes[i].set_ylim(df_system[col].min() + 0.05 * df_system[col].min(), 0.1)

            if fit:
                # Fit N for R > 3 Angstroms
                df_fit = df_system[df_system["R"] > system_starting_distance[i]].copy()
                if not df_fit.empty:
                    N = compute_N(df_fit, col, sign_flip=True, print_lvl=1)

                    # Generate fitted curve
                    R_range = np.linspace(min(df_system["R"]), max(df_system["R"]), 100)

                    # Find C coefficient using one point
                    ref_point = df_fit.iloc[0]
                    C = -ref_point[col] * (ref_point["R"] ** N)

                    # Calculate fitted values
                    fitted_values = -C / (R_range**N)

                    # Plot fitted curve
                    axes[i].plot(
                        R_range,
                        fitted_values,
                        color=color,
                        linestyle="--",
                        label=f"{labels[col]} (Fit, N={N:.1f})",
                    )

        # Set plot properties
        axes[i].set_title(f"{system.capitalize()} Dimer")
        axes[i].set_ylabel("Disp. Energy (kcal/mol)")
        axes[i].grid(True, linestyle="--", alpha=0.7)
        # minor ticks
        axes[i].minorticks_on()
        axes[i].tick_params(which="both", width=1)
        axes[i].legend()

        # Format tick labels to show actual values instead of powers
        axes[i].xaxis.set_major_formatter(ScalarFormatter())
        axes[i].yaxis.set_major_formatter(ScalarFormatter())

    axes[1].set_xlabel("Distance (Å)")
    plt.tight_layout()
    plt.savefig("dispersion_comparison_2.png", dpi=300)
    return


if __name__ == "__main__":
    # full()
    # hrcl_jobs.sqlt.table_to_df_pkl(
    #     db_p='curves.db',
    #     table='main',
    #     df_p="curves.pkl",
    # )
    plot_results(False)
