import pandas as pd
import matplotlib.pyplot as plt
from qcelemental import constants
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
import numpy as np
import psi4
from qm_tools_aw import tools
from pprint import pprint as pp

psi4.set_memory("32 GB")
psi4.set_num_threads(12)

h2kcalmol = constants.conversion_factor("hartree", "kcal/mol")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
    }
)


def plot_c6_extrapolation_charges(
    df_results,
    eq_distance,
    system_label="45_Ethyne-Pentane",
    output_dir="./plots/c6_extrapolation",
):
    """
    Plot C6 changes and dispersion energies vs distance.

    Uses similar plotting logic to subplot_all_curves_LoS_basis_set_D4_versions_nondamped()
    with three vertical subplots:
    - Top: C6 sum changes for monomers A and B
    - Middle: Dispersion energies (damped and non-damped)
    - Bottom: Partial charge changes for monomers A and B

    Parameters
    ----------
    df_results : pd.DataFrame
        Results from c6_change_mon_dimer_extrapolation()
    system_label : str
        System label for plot title
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    tick_fontsize = 14
    legend_fontsize = 12

    # Create 3x1 subplot
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), dpi=300)
    ax_top = axs[0]
    ax_mid = axs[1]
    ax_bot = axs[2]

    # Define colors
    colors = {
        # D4 C6 changes
        "C6_change_A_D4": "blue",
        "C6_change_B_D4": "red",
        # D3 C6 changes
        "C6_change_A_D3": "deepskyblue",
        "C6_change_B_D3": "salmon",
        # Partial charge changes (D4)
        "q_change_A_D4": "navy",
        "q_change_B_D4": "darkred",
        # Partial charge changes (MBIS)
        "q_change_A_MBIS": "dodgerblue",
        "q_change_B_MBIS": "lightcoral",
        # Dispersion energies
        "disp_nd": "orange",
        "disp_d": "teal",
        "disp_d_D3_C6s": "cyan",
        "disp_D3_supermolecular": "magenta",
        "delta_A": "purple",
        "delta_B": "green",
        "delta_A_BJ": "darkviolet",
        "delta_B_BJ": "darkgreen",
        "disp_hfd4_supermolecular": "brown",
        "sapt0/adz": "black",
    }
    markers = {
        "C6_change_A_D4": "o",
        "C6_change_B_D4": "s",
        "C6_change_A_D3": "^",
        "C6_change_B_D3": "v",
        "q_change_A_D4": "D",
        "q_change_B_D4": "X",
        "q_change_A_MBIS": "o",
        "q_change_B_MBIS": "s",
        "disp_nd": "s",
        "disp_d": "d",
        "disp_d_D3_C6s": "h",
        "disp_D3_supermolecular": "H",
        "delta_A": "v",
        "delta_B": "P",
        "delta_A_BJ": "<",
        "delta_B_BJ": ">",
        "disp_hfd4_supermolecular": "P",
        "sapt0/adz": "*",
    }

    # ===== TOP PLOT: C6 sum changes =====
    # D4 C6 changes
    ax_mid.plot(
        df_results["distance"],
        df_results["C6_sum_change_A_D4"],
        color=colors["C6_change_A_D4"],
        marker=markers["C6_change_A_D4"],
        label=r"$\Delta C_6^{AA}$ (-D4)",
        markersize=4,
    )
    ax_mid.plot(
        df_results["distance"],
        df_results["C6_sum_change_B_D4"],
        color=colors["C6_change_B_D4"],
        marker=markers["C6_change_B_D4"],
        label=r"$\Delta C_6^{BB}$ (-D4)",
        markersize=4,
    )
    # D3 C6 changes
    ax_mid.plot(
        df_results["distance"],
        df_results["C6_sum_change_A_D3"],
        color=colors["C6_change_A_D3"],
        marker=markers["C6_change_A_D3"],
        label=r"$\Delta C_6^{AA}$ (-D3)",
        markersize=4,
    )
    ax_mid.plot(
        df_results["distance"],
        df_results["C6_sum_change_B_D3"],
        color=colors["C6_change_B_D3"],
        marker=markers["C6_change_B_D3"],
        label=r"$\Delta C_6^{BB}$ (-D3)",
        markersize=4,
    )

    # Add horizontal line at 0
    ax_mid.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    # Add vertical line at equilibrium distance
    ax_mid.axvline(eq_distance, color="grey", linestyle="-", linewidth=1.0, alpha=0.6)

    ax_mid.text(
        -0.14,
        1.0,
        "(B)",
        transform=ax_mid.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_mid.set_ylabel(r"$\Delta C_6$ Sum (a.u.)")
    ax_mid.minorticks_on()
    ax_mid.tick_params(
        which="both",
        width=1,
        labelsize=tick_fontsize,
        direction="in",
        top=True,
        right=True,
    )
    ax_mid.legend(fontsize=legend_fontsize, loc="center right")
    ax_mid.xaxis.set_major_formatter(ScalarFormatter())
    ax_mid.yaxis.set_major_formatter(ScalarFormatter())

    # ===== MIDDLE PLOT: Dispersion energies =====
    # ax_mid.plot(
    #     df_results["distance"],
    #     df_results["disp_inter_no_damping"],
    #     color=colors["disp_nd"],
    #     marker=markers["disp_nd"],
    #     label="Intermolecular -D4 (ND)",
    #     markersize=5,
    # )
    ax_top.plot(
        df_results["distance"],
        df_results["disp_inter_damped"],
        color=colors["disp_d"],
        marker=markers["disp_d"],
        label="HF-D4(BJ) Intermolecular ",
        markersize=4,
    )
    ax_top.plot(
        df_results["distance"],
        df_results["disp_inter_damped_D3_C6s"],
        color=colors["disp_d_D3_C6s"],
        marker=markers["disp_d_D3_C6s"],
        label="HF-D3(BJ) Intermolecular",
        markersize=4,
    )
    ax_top.plot(
        df_results["distance"],
        df_results["disp_D3_supermolecular"],
        color=colors["disp_D3_supermolecular"],
        marker=markers["disp_D3_supermolecular"],
        label="HF-D3(BJ) Supermolecular IE",
        markersize=4,
    )
    ax_top.plot(
        df_results["distance"],
        df_results["disp_hfd4_supermolecular"],
        color=colors["disp_hfd4_supermolecular"],
        marker=markers["disp_hfd4_supermolecular"],
        label="HF-D4(BJ) Supermolecular IE",
        markersize=4,
    )
    # ax_mid.plot(
    #     df_results["distance"],
    #     df_results["disp_delta_A_no_damping"],
    #     color=colors["delta_A"],
    #     marker=markers["delta_A"],
    #     label=r"$\delta$ -D4 A (ND)",
    #     markersize=4,
    # )
    # ax_mid.plot(
    #     df_results["distance"],
    #     df_results["disp_delta_B_no_damping"],
    #     color=colors["delta_B"],
    #     marker=markers["delta_B"],
    #     label=r"$\delta$ -D4 B (ND)",
    #     markersize=4,
    # )
    ax_top.plot(
        df_results["distance"],
        df_results["disp_delta_A_damped"],
        color=colors["delta_A_BJ"],
        marker=markers["delta_A_BJ"],
        label=r"$\Delta^{AB}_{A}$ HF-D4(BJ)",
        markersize=4,
    )
    ax_top.plot(
        df_results["distance"],
        df_results["disp_delta_B_damped"],
        color=colors["delta_B_BJ"],
        marker=markers["delta_B_BJ"],
        label=r"$\Delta^{AB}_{B}$ HF-D4(BJ)",
        markersize=4,
    )
    ax_top.plot(
        df_results["distance"],
        df_results["sapt0/adz"],
        color=colors["sapt0/adz"],
        marker=markers["sapt0/adz"],
        label="SAPT0/aug-cc-pV(D+d)Z",
        markersize=4,
    )

    # Add horizontal line at 0
    ax_top.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    # Add vertical line at equilibrium distance
    ax_top.axvline(eq_distance, color="grey", linestyle="-", linewidth=1.0, alpha=0.6)

    # Create inset plot for close distances (zoomed view of small energies)
    # Focus on the last few points to highlight the smaller intermolecular energies
    # n_inset_points = min(8, len(df_results))
    n_inset_points = len(df_results) // 3  # Last third of points
    inset_data = df_results.iloc[n_inset_points:]

    ax_inset = inset_axes(
        ax_top,
        width="60%",
        height="35%",
        loc="center right",
        bbox_to_anchor=(0, 0.1, 1, 1),
        bbox_transform=ax_top.transAxes,
        borderpad=1.5,
    )

    # Plot only the key intermolecular dispersion curves in inset
    ax_inset.plot(
        inset_data["distance"],
        inset_data["disp_inter_no_damping"],
        color=colors["disp_nd"],
        marker=markers["disp_nd"],
        markersize=3,
        linewidth=1,
    )
    ax_inset.plot(
        inset_data["distance"],
        inset_data["disp_inter_damped"],
        color=colors["disp_d"],
        marker=markers["disp_d"],
        markersize=3,
        linewidth=1,
    )
    ax_inset.plot(
        inset_data["distance"],
        inset_data["disp_inter_damped_D3_C6s"],
        color=colors["disp_d_D3_C6s"],
        marker=markers["disp_d_D3_C6s"],
        markersize=3,
        linewidth=1,
    )
    ax_inset.plot(
        inset_data["distance"],
        inset_data["disp_D3_supermolecular"],
        color=colors["disp_D3_supermolecular"],
        marker=markers["disp_D3_supermolecular"],
        markersize=3,
        linewidth=1,
    )
    ax_inset.plot(
        inset_data["distance"],
        inset_data["disp_hfd4_supermolecular"],
        color=colors["disp_hfd4_supermolecular"],
        marker=markers["disp_hfd4_supermolecular"],
        markersize=3,
        linewidth=1,
    )
    ax_inset.plot(
        inset_data["distance"],
        inset_data["sapt0/adz"],
        color=colors["sapt0/adz"],
        marker=markers["sapt0/adz"],
        markersize=3,
        linewidth=1,
        label="SAPT0/aug-cc-pV(D+d)Z",
    )
    # delta -D4 terms
    ax_inset.plot(
        inset_data["distance"],
        inset_data["disp_delta_A_damped"],
        color=colors["delta_A_BJ"],
        marker=markers["delta_A_BJ"],
        markersize=3,
        linewidth=1,
    )
    ax_inset.plot(
        inset_data["distance"],
        inset_data["disp_delta_B_damped"],
        color=colors["delta_B_BJ"],
        marker=markers["delta_B_BJ"],
        markersize=3,
        linewidth=1,
    )

    ax_inset.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    # Add vertical line at equilibrium distance (if within inset range)
    if eq_distance <= inset_data["distance"].max():
        ax_inset.axvline(
            eq_distance, color="grey", linestyle="-", linewidth=0.8, alpha=0.6
        )
    ax_inset.tick_params(
        which="both",
        labelsize=8,
        direction="in",
        top=True,
        right=True,
    )
    ax_inset.set_xlabel(r"Distance (\AA)", fontsize=8)
    ax_inset.set_ylabel("Disp. Energy\n(kcal/mol)", fontsize=8)
    ax_inset.set_xlim(
        inset_data["distance"].min() - 0.2, inset_data["distance"].max() + 0.2
    )
    # ax_inset minor ticks
    ax_inset.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_inset.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Mark the inset region on the main plot
    mark_inset(ax_top, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.5)

    ax_top.text(
        -0.14,
        1.0,
        "(A)",
        transform=ax_top.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_top.set_ylabel("Disp. Energy (kcal/mol)")
    ax_top.minorticks_on()
    ax_top.tick_params(
        which="both",
        width=1,
        labelsize=tick_fontsize,
        direction="in",
        top=True,
        right=True,
    )
    ax_top.legend(fontsize=legend_fontsize - 4, loc="lower right", ncol=2)
    ax_top.xaxis.set_major_formatter(ScalarFormatter())
    ax_top.yaxis.set_major_formatter(ScalarFormatter())

    # ===== BOTTOM PLOT: Partial charge changes =====
    # D4 charges
    ax_bot.plot(
        df_results["distance"],
        df_results["qs_sum_change_A_D4"],
        color=colors["q_change_A_D4"],
        marker=markers["q_change_A_D4"],
        label=r"$\Delta q^{A}$ (-D4)",
        markersize=4,
    )
    ax_bot.plot(
        df_results["distance"],
        df_results["qs_sum_change_B_D4"],
        color=colors["q_change_B_D4"],
        marker=markers["q_change_B_D4"],
        label=r"$\Delta q^{B}$ (-D4)",
        markersize=4,
    )

    # MBIS charges (if available)
    if "qs_sum_change_A_MBIS" in df_results.columns:
        ax_bot.plot(
            df_results["distance"],
            df_results["qs_sum_change_A_MBIS"],
            color=colors["q_change_A_MBIS"],
            marker=markers["q_change_A_MBIS"],
            label=r"$\Delta q^{A}$ (MBIS)",
            markersize=4,
        )
        ax_bot.plot(
            df_results["distance"],
            df_results["qs_sum_change_B_MBIS"],
            color=colors["q_change_B_MBIS"],
            marker=markers["q_change_B_MBIS"],
            label=r"$\Delta q^{B}$ (MBIS)",
            markersize=4,
        )

    # Add horizontal line at 0
    ax_bot.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    # Add vertical line at equilibrium distance
    ax_bot.axvline(eq_distance, color="grey", linestyle="-", linewidth=1.0, alpha=0.6)

    ax_bot.text(
        -0.14,
        1.0,
        "(C)",
        transform=ax_bot.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_bot.set_xlabel(r"Distance (\AA)")
    ax_bot.set_ylabel(r"$\Delta q$ Sum (e)")
    ax_bot.minorticks_on()
    ax_bot.tick_params(
        which="both",
        width=1,
        labelsize=tick_fontsize,
        direction="in",
        top=True,
        right=True,
    )
    ax_bot.legend(fontsize=legend_fontsize, loc="center right")
    ax_bot.xaxis.set_major_formatter(ScalarFormatter())
    ax_bot.yaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()

    # Save figure
    safe_label = system_label.replace(" ", "_").replace("/", "_")
    output_path = os.path.join(output_dir, f"{safe_label}_c6_extrapolation.png")
    plt.savefig(output_path)
    print(f"{output_path}")
    plt.close()
    return output_path


def compute_mbis_charges(df):
    charges_A, charges_B = [], []
    charges_dimer = []
    for n, r in df.iterrows():
        print(n, r)
        geom_d = tools.generate_p4input_from_df(
            r["Geometry"], r["charges"], r["monAs"], r["monBs"], units="angstrom"
        )
        psi4.geometry(geom_d)
        e_dimer, wfn_dimer = psi4.energy("pbe0/aug-cc-pv(d+d)z", return_wfn=True)
        psi4.oeprop(wfn_dimer, "MBIS_CHARGES")
        charges = np.array(wfn_dimer.variable("MBIS CHARGES")).flatten()
        charges_dimer.append(charges)
        if n == 0:
            geom_A, geom_B = geom_d.split("--")
            geom_A += "\nunits angstrom"
            print(geom_A)
            psi4.geometry(geom_A)
            e_a, wfn_a = psi4.energy("pbe0/aug-cc-pv(d+d)z", return_wfn=True)
            psi4.oeprop(wfn_a, "MBIS_CHARGES")
            q_A = np.array(wfn_a.variable("MBIS CHARGES")).flatten()
            print("MBIS charges for monomer A:", charges)
            print(geom_B)
            psi4.geometry(geom_B)
            e_b, wfn_b = psi4.energy("pbe0/aug-cc-pv(d+d)z", return_wfn=True)
            psi4.oeprop(wfn_b, "MBIS_CHARGES")
            q_B = np.array(wfn_b.variable("MBIS CHARGES")).flatten()
            print("MBIS charges for monomer B:", charges)
        charges_A.append(q_A)
        charges_B.append(q_B)
    df["q_A"] = charges_A
    df["q_B"] = charges_B
    df["q_dimer"] = charges_dimer
    df.to_pickle(
        "./plots/c6_extrapolation/45_Ethyne-Pentane_c6_extrapolation_results_with_mbis.pkl"
    )
    return df


def compute_sapt_charges(df):
    sapt_disp = []
    for n, r in df.iterrows():
        print(n, r)
        geom_d = tools.generate_p4input_from_df(
            r["Geometry"], r["charges"], r["monAs"], r["monBs"], units="angstrom"
        )
        psi4.geometry(geom_d)
        # e_dimer, wfn_dimer = psi4.energy("sapt2+3(ccd)/aug-cc-pv(d+d)z", return_wfn=True)
        e_dimer = psi4.energy("sapt0/aug-cc-pv(d+d)z")
        qcvars = psi4.core.variables()
        pp(qcvars)
        sapt_disp.append(qcvars["SAPT DISP ENERGY"] * h2kcalmol)
        print("SAPT DISP:", sapt_disp[-1])
    df["sapt0/adz"] = sapt_disp
    df.to_pickle(
        "./plots/c6_extrapolation/45_Ethyne-Pentane_c6_extrapolation_results_with_sapt.pkl"
    )
    return df


def add_mbis_charge_changes(df):
    """
    Compute MBIS charge sum changes for monomers A and B.
    Assumes df has columns 'q_A' and 'q_B' containing MBIS charges.
    """
    if "q_A" not in df.columns or "q_B" not in df.columns:
        print("Warning: MBIS charges not found in dataframe")
        return df

    # Initialize lists for MBIS charge changes
    mbis_change_A = []
    mbis_change_B = []
    q_A_sum = np.sum(df.iloc[0]["q_A"])
    q_B_sum = np.sum(df.iloc[0]["q_B"])

    for idx, row in df.iterrows():
        # Sum of MBIS charges for monomers in current geometry
        q_dimer_sum_A = np.sum(
            row["q_dimer"][: len(row["q_A"])]
        )  # Charges for monomer A in dimer
        q_dimer_sum_B = np.sum(
            row["q_dimer"][len(row["q_A"]) :]
        )  # Charges for monomer B in dimer

        # Change relative to first (reference) geometry
        mbis_change_A.append(q_dimer_sum_A - q_A_sum)
        mbis_change_B.append(q_dimer_sum_B - q_B_sum)
        print(
            f"{idx}: MBIS d A = {mbis_change_A[-1]:.8f}, for B = {mbis_change_B[-1]:.8f}"
        )

    df["qs_sum_change_A_MBIS"] = mbis_change_A
    df["qs_sum_change_B_MBIS"] = mbis_change_B

    return df


def main():
    # df = pd.read_pickle(
    #     "./plots/c6_extrapolation/45_Ethyne-Pentane_c6_extrapolation_results.pkl"
    # )
    # compute_sapt_charges(df)
    # compute_mbis_charges(df)
    # df.to_pickle(
    #     "./plots/c6_extrapolation/45_Ethyne-Pentane_c6_extrapolation_results_with_mbis_with_sapt.pkl"
    # )
    # Reload dataframe with MBIS charges

    # Ensure that these are all dimer - monomer, says from monomer to dimer
    df = pd.read_pickle(
        "./plots/c6_extrapolation/45_Ethyne-Pentane_c6_extrapolation_results_with_mbis_with_sapt.pkl"
    )

    # Add MBIS charge change calculations
    df = add_mbis_charge_changes(df)
    plot_c6_extrapolation_charges(df, 3.14, system_label="45_Ethyne-Pentane")
    return


if __name__ == "__main__":
    main()
