import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
from src import paramsTable, locald4, plotting
from qm_tools_aw import tools
import os
from scipy.optimize import curve_fit
from src.plotting import prep_saptdft_components
from qcelemental import constants
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from src import dftd3

h2kcalmol = constants.conversion_factor("hartree", "kcal/mol")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
    }
)


def monomer_C6s_from_dimer(dimer_C6s, monA_C6s, monB_C6s):
    dimer_monA_C6s = dimer_C6s[: len(monA_C6s), : len(monA_C6s)]
    dimer_monB_C6s = dimer_C6s[len(monA_C6s) :, len(monA_C6s) :]
    intermolecular_C6s = dimer_C6s[: len(monA_C6s), len(monA_C6s) :] * 2
    return dimer_monA_C6s, dimer_monB_C6s, intermolecular_C6s


def c6_change_mon_dimer_plotting(df, system_label="45_Ethyne-Pentane", print_lvl=1):
    df_sys = df[df["System Label"] == system_label]
    df_sys.sort_values("distance (A)", inplace=True)
    pd.set_option("display.float_format", "{:.4f}".format)
    params = "sadz"
    params_no_damping = paramsTable.get_params(
        "SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING"
    )[0]
    print(params_no_damping)
    params = "SAPT_DFT_pbe0_adz_3_IE"
    print(f"Using params: {params}")
    if params == "sadz":
        params = paramsTable.get_params(params)
    else:
        params, _ = paramsTable.get_params(params)
    for n, r in df_sys[::-1].iterrows():
        print(f"System: {r['system_id']}, R: {r['R']}")
        dimer_C6s = r["C6s"]
        monA_C6s = r["C6_A"]
        monB_C6s = r["C6_B"]
        dimer_monA_C6s, dimer_monB_C6s, intermolecular_C6s = monomer_C6s_from_dimer(
            dimer_C6s, monA_C6s, monB_C6s
        )
        dimer_geom = r["Geometry_bohr"][:, 1:]
        distance_matrix = np.linalg.norm(dimer_geom[:, np.newaxis] - dimer_geom, axis=2)
        monomer_distance_A = distance_matrix[: len(monA_C6s), : len(monA_C6s)]
        monomer_distance_B = distance_matrix[len(monA_C6s) :, len(monA_C6s) :]
        intermolecular_distance = distance_matrix[: len(monA_C6s), len(monA_C6s) :]
        for i in range(len(monA_C6s)):
            monA_C6s[i, i] = 0.0
            dimer_monA_C6s[i, i] = 0.0
            monomer_distance_A[i, i] = 1.0
        for i in range(len(monB_C6s)):
            monB_C6s[i, i] = 0.0
            dimer_monB_C6s[i, i] = 0.0
            monomer_distance_B[i, i] = 1.0
        avg_change_A = np.mean(dimer_monA_C6s - monA_C6s)
        avg_change_B = np.mean(dimer_monB_C6s - monB_C6s)
        # average C6 percentage change - need to make diagonals 1 first to avoid div by 0
        dimer_monA_C6s_diag1 = dimer_monA_C6s.copy()
        dimer_monB_C6s_diag1 = dimer_monB_C6s.copy()
        monA_C6s_diag1 = monA_C6s.copy()
        monB_C6s_diag1 = monB_C6s.copy()
        for i in range(len(monA_C6s)):
            dimer_monA_C6s_diag1[i, i] = 1.0
            monA_C6s_diag1[i, i] = 1.0
        for i in range(len(monB_C6s)):
            dimer_monB_C6s_diag1[i, i] = 1.0
            monB_C6s_diag1[i, i] = 1.0
        avg_pct_change_A = (
            np.mean((dimer_monA_C6s_diag1 - monA_C6s_diag1) / monA_C6s_diag1) * 100.0
        )
        avg_pct_change_B = (
            np.mean((dimer_monB_C6s_diag1 - monB_C6s_diag1) / monB_C6s_diag1) * 100.0
        )
        c6_diff_A = dimer_monA_C6s - monA_C6s
        c6_diff_B = dimer_monB_C6s - monB_C6s
        avg_change_A = np.mean(c6_diff_A)  # * constants.hartree2kcalmol
        avg_change_B = np.mean(c6_diff_B)  # * constants.hartree2kcalmol
        # * constants.hartree2kcalmol
        mae_change_A = np.mean(np.abs(c6_diff_A))
        mae_change_B = np.mean(np.abs(c6_diff_B))  # * constants.hart
        min_change_A = np.min(c6_diff_A)  # * constants.hartree2kcalmol
        min_change_B = np.min(c6_diff_B)  # * constants.hartree2kcalmol
        max_change_A = np.max(c6_diff_A)  # * constants.hartree2kcalmol
        max_change_B = np.max(c6_diff_B)
        if False:
            print(
                f"C6s*h2km    , avg change A: {avg_change_A:.2f}, avg change B: {avg_change_B:.2f}"
            )
            print("divided by 1/r^6")
            avg_change_A_r6 = (
                np.mean(
                    dimer_monA_C6s / monomer_distance_A**6
                    - monA_C6s / monomer_distance_A**6
                )
                * constants.hartree2kcalmol
            )
            avg_change_B_r6 = (
                np.mean(
                    dimer_monB_C6s / monomer_distance_B**6
                    - monB_C6s / monomer_distance_B**6
                )
                * constants.hartree2kcalmol
            )
            avg_change_inter_r6 = (
                np.mean(intermolecular_C6s / intermolecular_distance**6)
                * constants.hartree2kcalmol
            )
            print(
                f"C6s*h2km/r^6, avg change A: {avg_change_A_r6:.2f}, avg change B: {avg_change_B_r6:.2f}"
            )
            print("divided by 1/r^8")
            avg_change_A_r8 = (
                np.mean(
                    dimer_monA_C6s / monomer_distance_A**8
                    - monA_C6s / monomer_distance_A**8
                )
                * constants.hartree2kcalmol
            )
            avg_change_B_r8 = (
                np.mean(
                    dimer_monB_C6s / monomer_distance_B**8
                    - monB_C6s / monomer_distance_B**8
                )
                * constants.hartree2kcalmol
            )
            avgerage_inter_r8 = (
                np.mean(intermolecular_C6s / intermolecular_distance**8)
                * constants.hartree2kcalmol
            )
            print(
                f"C6s*h2km/r^8, avg change A: {avg_change_A_r8:.2f}, avg change B: {avg_change_B_r8:.2f}"
            )

        dimer_dispersion_supra = locald4.compute_disp_2B_supra_from_C6s(
            r["Geometry_bohr"][:, 0],
            r["Geometry_bohr"][:, 1:],
            dimer_C6s,
            r["monAs"],
            r["monBs"],
            params,
        )
        dimer_dispersion_supra_no_damping = (
            locald4.compute_disp_2B_supra_from_C6s_NO_DAMPING(
                r["Geometry_bohr"][:, 0],
                r["Geometry_bohr"][:, 1:],
                dimer_C6s,
                r["monAs"],
                r["monBs"],
                params_no_damping,
            )
        )
        dimer_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][:, 0],
            r["Geometry_bohr"][:, 1:],
            dimer_C6s,
            params_no_damping,
        )
        monA_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][: len(monA_C6s), 0],
            r["Geometry_bohr"][: len(monA_C6s), 1:],
            monA_C6s,
            params_no_damping,
        )
        monB_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][len(monA_C6s) :, 0],
            r["Geometry_bohr"][len(monA_C6s) :, 1:],
            monB_C6s,
            params_no_damping,
        )
        dimer_monA_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][: len(monA_C6s), 0],
            r["Geometry_bohr"][: len(monA_C6s), 1:],
            dimer_monA_C6s,
            params_no_damping,
        )
        dimer_monB_dispersion = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
            r["Geometry_bohr"][len(monA_C6s) :, 0],
            r["Geometry_bohr"][len(monA_C6s) :, 1:],
            dimer_monB_C6s,
            params_no_damping,
        )

        # Intermolecular contributions of no_damping

        bj = True
        non_damping = True
        print("===============  NOTE  ==================")
        print("dmonA/dmonB compute using dimer C6s subblocks")
        print("monA/monB compute using monomer C6s")
        print("change in C6s is defined as: dimer_monX_C6s - monX_C6s, X=A,B")
        print("=========================================")
        c6_sum_A = np.sum(monA_C6s)
        c6_sum_B = np.sum(monB_C6s)
        print(f"C6s sum, monA: {c6_sum_A:.2f}, monB: {c6_sum_B:.2f}")
        c6_sum_dimer_monA = np.sum(dimer_monA_C6s)
        c6_sum_dimer_monB = np.sum(dimer_monB_C6s)
        print(
            f"C6s sum, dmonA: {c6_sum_dimer_monA:.2f}, dmonB: {c6_sum_dimer_monB:.2f}"
        )
        print(
            f"C6s sum change A: {c6_sum_dimer_monA - c6_sum_A:.2f}, change B: {c6_sum_dimer_monB - c6_sum_B:.2f}"
        )
        print(
            f"C6s, avg change A: {avg_change_A:.2f}, avg change B: {avg_change_B:.2f}"
        )
        print(
            f"C6s, mae change A: {mae_change_A:.2f}, mae change B: {mae_change_B:.2f}"
        )
        print(
            f"C6s, min change A: {min_change_A:.2f}, min change B: {min_change_B:.2f}"
        )
        print(
            f"C6s, max change A: {max_change_A:.2f}, max change B: {max_change_B:.2f}"
        )
        print(
            f"C6s pct chg, avg pct change A: {avg_pct_change_A:.2f}%, avg pct change B: {avg_pct_change_B:.2f}%"
        )
        if non_damping:
            print("----- NO DAMPING -----")
            print(
                f"Disp.    dimer: {dimer_dispersion:.4f}, monA: {monA_dispersion:.4f}, monB: {monB_dispersion:.4f}"
            )
            print(
                f"Disp.    Total: {dimer_dispersion - monA_dispersion - monB_dispersion:.4f}"
            )
            # print("    Using only C6s from the MONOMERS for intramolecular pairs (should agree with supra)")
            # print(
            #     f"Disp.MC6 dimer: {dimer_dispersion_c6s_mon:.4f}, monA: {monA_dispersion:.4f}, monB: {monB_dispersion:.4f}"
            # )
            # print(
            #     f"Disp.MC6 Total: {dimer_dispersion_c6s_mon - monA_dispersion - monB_dispersion:.4f}"
            # )
            print(
                f"Disp.    dmonA: {dimer_monA_dispersion:.4f}, dmonB: {dimer_monB_dispersion:.4f}"
            )
            print(
                f"Disp.     monA: {monA_dispersion:.4f},  monB: {monB_dispersion:.4f}"
            )
            # Difference between monA_dispersion and dimer_monA_dispersion
            monA_diff = dimer_monA_dispersion - monA_dispersion
            monB_diff = dimer_monB_dispersion - monB_dispersion
            print(f"Disp.  delta A: {monA_diff:.4f}, delta B: {monB_diff:.4f}")
            # Sum changes
            print(f"Disp.  delta A+B: {monA_diff + monB_diff:.4f}")
            print(f"Disp.  Intermolecular: {dimer_dispersion_supra_no_damping:.4f}")
            print(
                f"Disp.  delta A+B+Intermolecular: {monA_diff + monB_diff + dimer_dispersion_supra_no_damping:.4f}"
            )
            print(f"{len(monA_C6s) = }, {len(monB_C6s) = }")
        if bj:
            print("----- BJ DAMPING -----")
            # Now for BJ
            dimer_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][:, 0], r["Geometry_bohr"][:, 1:], dimer_C6s, params
            )
            monA_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][: len(monA_C6s), 0],
                r["Geometry_bohr"][: len(monA_C6s), 1:],
                monA_C6s,
                params,
            )
            monB_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][len(monA_C6s) :, 0],
                r["Geometry_bohr"][len(monA_C6s) :, 1:],
                monB_C6s,
                params,
            )
            dimer_monA_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][: len(monA_C6s), 0],
                r["Geometry_bohr"][: len(monA_C6s), 1:],
                dimer_monA_C6s,
                params,
            )
            dimer_monB_dispersion_BJ = locald4.compute_disp_2B_from_C6s(
                r["Geometry_bohr"][len(monA_C6s) :, 0],
                r["Geometry_bohr"][len(monA_C6s) :, 1:],
                dimer_monB_C6s,
                params,
            )
            print(
                f"Disp.BJ dimer: {dimer_dispersion_BJ:.4f}, monA: {monA_dispersion_BJ:.4f}, monB: {monB_dispersion_BJ:.4f}"
            )
            print(
                f"Disp.BJ Total: {dimer_dispersion_BJ - monA_dispersion_BJ - monB_dispersion_BJ:.4f}"
            )
            print(
                f"Disp.BJ  dmonA: {dimer_monA_dispersion_BJ:.4f}, dmonB: {dimer_monB_dispersion_BJ:.4f}"
            )
            print(
                f"Disp.BJ   monA: {monA_dispersion_BJ:.4f},  monB: {monB_dispersion_BJ:.4f}"
            )
            delta_monA_BJ = dimer_monA_dispersion_BJ - monA_dispersion_BJ
            delta_monB_BJ = dimer_monB_dispersion_BJ - monB_dispersion_BJ
            print(f"Disp.BJ delta A: {delta_monA_BJ:.4f}, delta B: {delta_monB_BJ:.4f}")
            print(f"Disp.BJ delta A+B: {delta_monA_BJ + delta_monB_BJ:.4f}")
            print(f"Disp.BJ Intermolecular: {dimer_dispersion_supra:.4f}")
            print(
                f"Disp.BJ delta A+B+Intermolecular: {delta_monA_BJ + delta_monB_BJ + dimer_dispersion_supra:.4f}"
            )
            print(f"{len(monA_C6s) = }, {len(monB_C6s) = }")

        return
        if print_lvl == 0:
            params_damped, _ = paramsTable.param_lookup("sadz")
            params_undamped = [1.0, 1.0, 0.0, 0.0]
            t6_2, t8_2, energies = locald4.compute_bj_terms(
                dimer_geom[:, 0],
                dimer_geom[:, 1:],
                dimer_C6s,
                params=params_damped,
                # params=params_undamped,
                damping_2d=True,
            )
            print(sum(energies))
            print("dimer C6s")
            print(dimer_C6s)
            print("Monomer A C6s")
            print(monA_C6s)
            print("Dimer Monomer A C6s")
            print(dimer_monA_C6s)
            print("Monomer B C6s")
            print(monB_C6s)
            print("Dimer Monomer B C6s")
            print(dimer_monB_C6s)
            print("C6 change")
            print(dimer_monA_C6s - monA_C6s)
            print(dimer_monB_C6s - monB_C6s)
            monA_t6s = t6_2[: len(monA_C6s), : len(monA_C6s)]
            monB_t6s = t6_2[len(monA_C6s) :, len(monA_C6s) :]
            monA_t8s = t8_2[: len(monA_C6s), : len(monA_C6s)]
            monB_t8s = t8_2[len(monA_C6s) :, len(monA_C6s) :]
            print("Dimer t6s")
            print(t6_2)
            print("Monomer t6s")
            print(monA_t6s)
            print(monB_t6s)
            print("Monomer t8s")
            print(monA_t8s)
            print(monB_t8s)
        print()
    return


def c6_change_mon_dimer_extrapolation(
    df,
    system_label="45_Ethyne-Pentane",
    step_size=0.5,
    upper_boundary=20.0,
    starting_distance=None,
    print_lvl=1,
):
    """
    Analyze how C6 coefficients change as monomers are pulled apart along the
    original displacement vector.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing geometry and monomer information
    system_label : str
        System label to analyze (e.g., "45_Ethyne-Pentane")
    step_size : float
        Distance step size in Angstroms for extrapolation
    upper_boundary : float
        Maximum distance to extrapolate to in Angstroms
    starting_distance : float, optional
        Starting distance; if None, uses the equilibrium distance from data
    print_lvl : int
        Print level for debugging output

    Returns
    -------
    df_results : pd.DataFrame
        DataFrame containing C6 changes and dispersion energies at each distance
    """
    df_sys = df[df["System Label"] == system_label].copy()
    df_sys = df_sys.sort_values("distance (A)")

    if len(df_sys) < 2:
        raise ValueError(f"Need at least 2 geometries for {system_label}")

    # Get parameters
    params_no_damping = paramsTable.get_params(
        "SAPT_DFT_pbe0_adz_3_IE_supra_NO_DAMPING"
    )[0]
    params_damped, _ = paramsTable.get_params("SAPT_DFT_pbe0_adz_3_IE")

    # Get two geometries to compute displacement vector
    # Use the two closest points
    r1 = df_sys.iloc[0]
    r2 = df_sys.iloc[1]

    geom1 = r1["Geometry"][:, 1:]  # xyz coordinates in Angstroms
    geom2 = r2["Geometry"][:, 1:]
    atom_numbers = r1["Geometry"][:, 0].astype(int)
    monAs = r1["monAs"]
    monBs = r1["monBs"]
    charges = r1["charges"] if "charges" in r1.index else [[0, 1], [0, 1], [0, 1]]

    # Compute center of mass for each monomer in both geometries
    def com(geom, indices):
        return np.mean(geom[indices], axis=0)

    com_A1 = com(geom1, monAs)
    com_B1 = com(geom1, monBs)
    com_A2 = com(geom2, monAs)
    com_B2 = com(geom2, monBs)

    # Displacement vector: direction monomer B moves relative to A
    # between the two geometries
    disp_vec = (com_B2 - com_A2) - (com_B1 - com_A1)
    if np.linalg.norm(disp_vec) < 1e-8:
        # If monomers move together, use the intermolecular axis
        disp_vec = com_B1 - com_A1
    disp_vec = disp_vec / np.linalg.norm(disp_vec)  # Normalize

    if print_lvl > 0:
        print(f"System: {system_label}")
        print(f"Displacement vector: {disp_vec}")
        print(f"Step size: {step_size} Å, Upper boundary: {upper_boundary} Å")

    # Starting geometry (use equilibrium or first geometry)
    if starting_distance is None:
        starting_distance = r1["distance (A)"]

    base_geom = geom1.copy()
    base_geom_bohr = base_geom * constants.conversion_factor("angstrom", "bohr")

    # Generate distances to evaluate
    distances = np.arange(starting_distance, upper_boundary + step_size, step_size)

    # Storage for results
    results = {
        "distance": [],
        # D4 C6 sums
        "C6_sum_dimer_D4": [],
        "C6_sum_monA_D4": [],
        "C6_sum_monB_D4": [],
        "C6_sum_change_A_D4": [],
        "C6_sum_change_B_D4": [],
        # D3 C6 sums
        "C6_sum_dimer_D3": [],
        "C6_sum_monA_D3": [],
        "C6_sum_monB_D3": [],
        "C6_sum_change_A_D3": [],
        "C6_sum_change_B_D3": [],
        # Dispersion energies
        "disp_inter_no_damping": [],
        "disp_inter_damped": [],
        "disp_delta_A_no_damping": [],
        "disp_delta_B_no_damping": [],
        "disp_delta_A_damped": [],
        "disp_delta_B_damped": [],
    }

    # Compute C6s at each distance
    for dist in distances:
        # Shift monomer B along displacement vector
        shift = dist - starting_distance
        new_geom = base_geom.copy()
        new_geom[monBs] = new_geom[monBs] + shift * disp_vec

        # Convert to bohr for dftd4
        new_geom_bohr = new_geom * constants.conversion_factor("angstrom", "bohr")

        # Split geometries
        geom_A = new_geom[monAs]
        geom_B = new_geom[monBs]
        atom_A = atom_numbers[monAs]
        atom_B = atom_numbers[monBs]

        try:
            # Compute C6s for dimer and monomers
            C6s_dimer, C6s_mA, C6s_mB = locald4.calc_dftd4_c6_for_d_a_b(
                new_geom,  # dimer coords (Angstrom)
                atom_numbers,  # dimer atom numbers
                atom_A,  # monA atom numbers
                geom_A,  # monA coords
                atom_B,  # monB atom numbers
                geom_B,  # monB coords
                charges,
                dftd4_bin="dftd4",
            )

            # Extract dimer monomer C6 subblocks
            dimer_monA_C6s, dimer_monB_C6s, _ = monomer_C6s_from_dimer(
                C6s_dimer, C6s_mA, C6s_mB
            )

            # Compute C6 sums (D4)
            c6_sum_dimer = np.sum(C6s_dimer)
            c6_sum_monA = np.sum(C6s_mA)
            c6_sum_monB = np.sum(C6s_mB)
            c6_sum_dimer_monA = np.sum(dimer_monA_C6s)
            c6_sum_dimer_monB = np.sum(dimer_monB_C6s)

            # Collect D3 C6 coefficients (uses Angstrom coords)
            d3bin = "./simple-dftd3/_build/app/s-dftd3"
            d3_dimer_data, _, d3_dimer = dftd3.collect_bjm_d3data(
                atom_numbers, new_geom, ATM=False,s_dftd3_bin=d3bin
            )
            d3_monA_data, _, d3_monA = dftd3.collect_bjm_d3data(atom_A, geom_A, ATM=False, s_dftd3_bin=d3bin)
            d3_monB_data, _, d3_monB = dftd3.collect_bjm_d3data(atom_B, geom_B, ATM=False, s_dftd3_bin=d3bin)

            # Extract D3 C6 matrices
            C6s_dimer_D3 = d3_dimer_data["c6s"]
            C6s_mA_D3 = d3_monA_data["c6s"]
            C6s_mB_D3 = d3_monB_data["c6s"]

            # Extract dimer monomer C6 subblocks (D3)
            dimer_monA_C6s_D3, dimer_monB_C6s_D3, _ = monomer_C6s_from_dimer(
                C6s_dimer_D3, C6s_mA_D3, C6s_mB_D3
            )

            # Compute D3 C6 sums
            c6_sum_dimer_D3 = np.sum(C6s_dimer_D3)
            c6_sum_monA_D3 = np.sum(C6s_mA_D3)
            c6_sum_monB_D3 = np.sum(C6s_mB_D3)
            c6_sum_dimer_monA_D3 = np.sum(dimer_monA_C6s_D3)
            c6_sum_dimer_monB_D3 = np.sum(dimer_monB_C6s_D3)

            # Compute dispersion energies
            disp_inter_nd = locald4.compute_disp_2B_supra_from_C6s_NO_DAMPING(
                atom_numbers,
                new_geom_bohr,
                C6s_dimer,
                monAs,
                monBs,
                params_no_damping,
            )
            disp_inter_d = locald4.compute_disp_2B_supra_from_C6s(
                atom_numbers,
                new_geom_bohr,
                C6s_dimer,
                monAs,
                monBs,
                params_damped,
            )

            # Compute intramolecular dispersion changes
            monA_disp = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
                atom_A, new_geom_bohr[monAs], C6s_mA, params_no_damping
            )
            monB_disp = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
                atom_B, new_geom_bohr[monBs], C6s_mB, params_no_damping
            )
            dimer_monA_disp = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
                atom_A, new_geom_bohr[monAs], dimer_monA_C6s, params_no_damping
            )
            dimer_monB_disp = locald4.compute_disp_2B_from_C6s_NO_DAMPING(
                atom_B, new_geom_bohr[monBs], dimer_monB_C6s, params_no_damping
            )

            delta_A = dimer_monA_disp - monA_disp
            delta_B = dimer_monB_disp - monB_disp

            # Compute BJ-damped intramolecular dispersion changes
            monA_disp_BJ = locald4.compute_disp_2B_from_C6s(
                atom_A, new_geom_bohr[monAs], C6s_mA, params_damped
            )
            monB_disp_BJ = locald4.compute_disp_2B_from_C6s(
                atom_B, new_geom_bohr[monBs], C6s_mB, params_damped
            )
            dimer_monA_disp_BJ = locald4.compute_disp_2B_from_C6s(
                atom_A, new_geom_bohr[monAs], dimer_monA_C6s, params_damped
            )
            dimer_monB_disp_BJ = locald4.compute_disp_2B_from_C6s(
                atom_B, new_geom_bohr[monBs], dimer_monB_C6s, params_damped
            )
            delta_A_BJ = dimer_monA_disp_BJ - monA_disp_BJ
            delta_B_BJ = dimer_monB_disp_BJ - monB_disp_BJ

            # Store results
            results["distance"].append(dist)
            # D4 C6 results
            results["C6_sum_dimer_D4"].append(c6_sum_dimer)
            results["C6_sum_monA_D4"].append(c6_sum_monA)
            results["C6_sum_monB_D4"].append(c6_sum_monB)
            results["C6_sum_change_A_D4"].append(c6_sum_dimer_monA - c6_sum_monA)
            results["C6_sum_change_B_D4"].append(c6_sum_dimer_monB - c6_sum_monB)
            # D3 C6 results
            results["C6_sum_dimer_D3"].append(c6_sum_dimer_D3)
            results["C6_sum_monA_D3"].append(c6_sum_monA_D3)
            results["C6_sum_monB_D3"].append(c6_sum_monB_D3)
            results["C6_sum_change_A_D3"].append(c6_sum_dimer_monA_D3 - c6_sum_monA_D3)
            results["C6_sum_change_B_D3"].append(c6_sum_dimer_monB_D3 - c6_sum_monB_D3)
            # Dispersion energies
            results["disp_inter_no_damping"].append(disp_inter_nd)
            results["disp_inter_damped"].append(disp_inter_d)
            results["disp_delta_A_no_damping"].append(delta_A)
            results["disp_delta_B_no_damping"].append(delta_B)
            results["disp_delta_A_damped"].append(delta_A_BJ)
            results["disp_delta_B_damped"].append(delta_B_BJ)

            if print_lvl > 1:
                print(
                    f"d={dist:.2f} Å: C6 change A={c6_sum_dimer_monA - c6_sum_monA:.2f}, "
                    f"B={c6_sum_dimer_monB - c6_sum_monB:.2f}, "
                    f"Disp(ND)={disp_inter_nd:.4f}"
                )

        except Exception as e:
            if print_lvl > 0:
                print(f"Error at distance {dist:.2f}: {e}")
            continue

    df_results = pd.DataFrame(results)

    if print_lvl > 0:
        print(f"\nGenerated {len(df_results)} data points")
        print(df_results.head(10))

    return df_results


def plot_c6_extrapolation(
    df_results,
    system_label="45_Ethyne-Pentane",
    output_dir="./plots/c6_extrapolation",
):
    """
    Plot C6 changes and dispersion energies vs distance.

    Uses similar plotting logic to subplot_all_curves_LoS_basis_set_D4_versions_nondamped()
    with two vertical subplots:
    - Top: C6 sum changes for monomers A and B
    - Bottom: Dispersion energies (damped and non-damped)

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

    # Create 2x1 subplot
    fig, axs = plt.subplots(2, 1, figsize=(6, 7), dpi=300)
    ax_top = axs[0]
    ax_bot = axs[1]

    # Define colors
    colors = {
        # D4 C6 changes
        "C6_change_A_D4": "blue",
        "C6_change_B_D4": "red",
        # D3 C6 changes
        "C6_change_A_D3": "deepskyblue",
        "C6_change_B_D3": "salmon",
        # Dispersion energies
        "disp_nd": "orange",
        "disp_d": "teal",
        "delta_A": "purple",
        "delta_B": "green",
        "delta_A_BJ": "darkviolet",
        "delta_B_BJ": "darkgreen",
    }
    markers = {
        "C6_change_A_D4": "o",
        "C6_change_B_D4": "s",
        "C6_change_A_D3": "^",
        "C6_change_B_D3": "v",
        "disp_nd": "^",
        "disp_d": "d",
        "delta_A": "v",
        "delta_B": "P",
        "delta_A_BJ": "<",
        "delta_B_BJ": ">",
    }

    # ===== TOP PLOT: C6 sum changes =====
    # D4 C6 changes
    ax_top.plot(
        df_results["distance"],
        df_results["C6_sum_change_A_D4"],
        color=colors["C6_change_A_D4"],
        marker=markers["C6_change_A_D4"],
        label=r"$\Delta C_6^{AA}$ (D4)",
        markersize=4,
    )
    ax_top.plot(
        df_results["distance"],
        df_results["C6_sum_change_B_D4"],
        color=colors["C6_change_B_D4"],
        marker=markers["C6_change_B_D4"],
        label=r"$\Delta C_6^{BB}$ (D4)",
        markersize=4,
    )
    # D3 C6 changes
    ax_top.plot(
        df_results["distance"],
        df_results["C6_sum_change_A_D3"],
        color=colors["C6_change_A_D3"],
        marker=markers["C6_change_A_D3"],
        label=r"$\Delta C_6^{AA}$ (D3)",
        markersize=4,
    )
    ax_top.plot(
        df_results["distance"],
        df_results["C6_sum_change_B_D3"],
        color=colors["C6_change_B_D3"],
        marker=markers["C6_change_B_D3"],
        label=r"$\Delta C_6^{BB}$ (D3)",
        markersize=4,
    )

    # Add horizontal line at 0
    ax_top.axhline(0, color="grey", linestyle="--", linewidth=0.8)

    ax_top.text(
        -0.12,
        1.0,
        "(A)",
        transform=ax_top.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_top.set_ylabel(r"$\Delta C_6$ Sum (a.u.)")
    ax_top.minorticks_on()
    ax_top.tick_params(
        which="both",
        width=1,
        labelsize=tick_fontsize,
        direction="in",
        top=True,
        right=True,
    )
    ax_top.legend(fontsize=legend_fontsize, loc="upper right")
    ax_top.xaxis.set_major_formatter(ScalarFormatter())
    ax_top.yaxis.set_major_formatter(ScalarFormatter())

    # ===== BOTTOM PLOT: Dispersion energies =====
    ax_bot.plot(
        df_results["distance"],
        df_results["disp_inter_no_damping"],
        color=colors["disp_nd"],
        marker=markers["disp_nd"],
        label="Intermolecular -D4 (ND)",
        markersize=4,
    )
    ax_bot.plot(
        df_results["distance"],
        df_results["disp_inter_damped"],
        color=colors["disp_d"],
        marker=markers["disp_d"],
        label="Intermolecular -D4 (Damped)",
        markersize=4,
    )
    ax_bot.plot(
        df_results["distance"],
        df_results["disp_delta_A_no_damping"],
        color=colors["delta_A"],
        marker=markers["delta_A"],
        label=r"$\delta$ -D4 A (ND)",
        markersize=4,
    )
    ax_bot.plot(
        df_results["distance"],
        df_results["disp_delta_B_no_damping"],
        color=colors["delta_B"],
        marker=markers["delta_B"],
        label=r"$\delta$ -D4 B (ND)",
        markersize=4,
    )
    ax_bot.plot(
        df_results["distance"],
        df_results["disp_delta_A_damped"],
        color=colors["delta_A_BJ"],
        marker=markers["delta_A_BJ"],
        label=r"$\delta$ -D4 A (BJ)",
        markersize=4,
    )
    ax_bot.plot(
        df_results["distance"],
        df_results["disp_delta_B_damped"],
        color=colors["delta_B_BJ"],
        marker=markers["delta_B_BJ"],
        label=r"$\delta$ -D4 B (BJ)",
        markersize=4,
    )

    # Add horizontal line at 0
    ax_bot.axhline(0, color="grey", linestyle="--", linewidth=0.8)

    ax_bot.text(
        -0.12,
        1.0,
        "(B)",
        transform=ax_bot.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_bot.set_xlabel(r"Distance (\AA)")
    ax_bot.set_ylabel("Disp. Energy (kcal/mol)")
    ax_bot.minorticks_on()
    ax_bot.tick_params(
        which="both",
        width=1,
        labelsize=tick_fontsize,
        direction="in",
        top=True,
        right=True,
    )
    ax_bot.legend(fontsize=legend_fontsize - 2, loc="upper right")
    ax_bot.xaxis.set_major_formatter(ScalarFormatter())
    ax_bot.yaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()

    # Save figure
    safe_label = system_label.replace(" ", "_").replace("/", "_")
    output_path = os.path.join(output_dir, f"{safe_label}_c6_extrapolation.pdf")
    plt.savefig(output_path)
    print(f"{output_path}")
    plt.close()

    return output_path


def main():
    df = pd.read_pickle("./plots/ddft_curves.pkl")
    df_results = c6_change_mon_dimer_extrapolation(
        df,
        system_label="45_Ethyne-Pentane",
        step_size=1.0,
        upper_boundary=50.0,
    )
    plot_c6_extrapolation(df_results, system_label="45_Ethyne-Pentane")
    return


if __name__ == "__main__":
    main()
