import numpy as np
from periodictable import elements
import qcelemental as qcel
import subprocess
import os
import json
from . import r4r2
from .constants import Constants
from qcelemental import constants
from qm_tools_aw import tools

hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")


def write_xyz_from_np(atom_numbers, carts, outfile="dat.xyz", charges=[0, 1]) -> None:
    """
    write_xyz_from_np
    """
    with open(outfile, "w") as f:
        f.write(str(len(carts)) + "\n\n")
        for n, i in enumerate(carts):
            el = str(int(atom_numbers[n]))
            v = "    ".join(["%.16f" % k for k in i])
            line = "%s    %s\n" % (el, v)
            f.write(line)
    return


def get_monomer_C6s_from_dimer(C6s_dimer, monN) -> np.array:
    C6s_monomer_from_dimer = C6s_dimer[monN].tolist()
    for i in range(len(C6s_monomer_from_dimer)):
        t1 = C6s_monomer_from_dimer[i]
        t2 = []
        for a in monN:
            t2.append(t1[a])
        C6s_monomer_from_dimer[i] = t2
    C6s_monomer_from_dimer = np.array(C6s_monomer_from_dimer)
    return C6s_monomer_from_dimer


def calc_dftd4_c6_c8_pairDisp2(
    atom_numbers: np.array,
    carts: np.array,
    charges: np.array,
    input_xyz: str = "dat.xyz",
    dftd4_bin: str = "/theoryfs2/ds/amwalla3/.local/bin/dftd4",
    p: [] = [1.0, 1.61679827, 0.44959224, 3.35743605],
    s9=0.0,
):
    """
    Ensure that dftd4 binary is from compiling git@github.com:Awallace3/dftd4
        - this is used to generate more decimal places on values for c6, c8,
          and pairDisp2
    """

    write_xyz_from_np(
        atom_numbers,
        carts,
        outfile=input_xyz,
        charges=charges,
    )
    args = [
        dftd4_bin,
        input_xyz,
        "--property",
        "--param",
        str(p[0]),
        str(p[1]),
        str(p[2]),
        str(p[3]),
        "--mbdscale",
        f"{s9}",
        "-c",
        str(charges[0]),
        "--pair-resolved",
    ]
    # print(" ".join(args))
    subprocess.call(
        # cmd,
        # shell=True,
        args=args,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    output_json = "C_n.json"
    with open(output_json) as f:
        cs = json.load(f)
    C6s = np.array(cs["c6"], dtype=np.float64)
    C8s = np.array(cs["c8"], dtype=np.float64)
    output_json = "pairs.json"
    with open(output_json) as f:
        pairs = json.load(f)
        pairs = np.array(pairs["pairs2"])
    with open(".EDISP", "r") as f:
        e = float(f.read())
    os.remove(input_xyz)
    os.remove("C_n.json")
    os.remove("pairs.json")
    return C6s, C8s, pairs, e


def calc_dftd4_c6_for_d_a_b(
    cD,
    pD,
    pA,
    cA,
    pB,
    cB,
    charges: np.array,
    input_xyz: str = "dat.xyz",
    dftd4_bin: str = "/theoryfs2/ds/amwalla3/.local/bin/dftd4",
    p: [] = [1.0, 1.61679827, 0.44959224, 3.35743605],
    s9=0.0,
):
    C6s_dimer, _, _, df_c_e = calc_dftd4_c6_c8_pairDisp2(
        pD,
        cD,
        charges[0],
        p=p,
    )
    C6s_mA, _, _, _ = calc_dftd4_c6_c8_pairDisp2(
        pA,
        cA,
        charges[1],
        p=p,
    )
    C6s_mB, _, _, _ = calc_dftd4_c6_c8_pairDisp2(
        pB,
        cB,
        charges[2],
        p=p,
    )
    return C6s_dimer, C6s_mA, C6s_mB


def read_EDISP() -> None:
    """
    read_EDISP returns dftd4 .EDISP value
    """
    with open(".EDISP", "r") as f:
        return float(f.read())


def compute_bj_f90(
    pos: np.array,
    carts: np.array,
    C6s: np.array,
    params: [] = [1.61679827, 0.44959224, 3.35743605],
    r4r2_ls=r4r2.r4r2_vals_ls(),
) -> float:
    """
    compute_bj_f90 computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    energy = 0
    if len(params) == 3:
        s8, a1, a2 = params
        s6 = 1.0
    elif len(params) == 4:
        s6, s8, a1, a2 = params
    else:
        raise ValueError("params must be length 3 or 4")
    M_tot = len(carts)
    energies = np.zeros(M_tot)
    lattice_points = 1

    for i in range(M_tot):
        el1 = int(pos[i])
        Q_A = (0.5 * el1**0.5 * r4r2_ls[el1 - 1]) ** 0.5

        for j in range(i + 1):
            el2 = int(pos[j])
            Q_B = (0.5 * el2**0.5 * r4r2_ls[el2 - 1]) ** 0.5
            if i == j:
                continue
            for k in range(lattice_points):
                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                C6ij = C6s[i, j]

                r1, r2 = carts[i, :], carts[j, :]
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))

                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)

                edisp = s6 * t6 + s8 * rrij * t8

                de = -C6ij * edisp * 0.5
                # print(i + 1, j + 1, r2, r0ij, edisp, de)
                energies[i] += de
                if i != j:
                    energies[j] += de
    energy = np.sum(energies)
    return energy


def compute_bj_f90_ATM(
    pos: np.array,
    carts: np.array,
    C6s: np.array,
    params: [] = [1.61679827, 0.44959224, 3.35743605, 1.0],
    # [s6, s8, a1, a2, s9]
    r4r2_ls=r4r2.r4r2_vals_ls(),
) -> float:
    """
    compute_bj_f90 computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    energy = 0
    s6, s8, a1, a2, s9 = params
    M_tot = len(carts)
    energies = np.zeros(M_tot)
    lattice_points = 1

    for i in range(M_tot):
        el1 = int(pos[i])
        Q_A = (0.5 * el1**0.5 * r4r2_ls[el1 - 1]) ** 0.5

        for j in range(i + 1):
            el2 = int(pos[j])
            Q_B = (0.5 * el2**0.5 * r4r2_ls[el2 - 1]) ** 0.5
            if i == j:
                continue
            # for
            for k in range(lattice_points):
                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                C6ij = C6s[i, j]

                r1, r2 = carts[i, :], carts[j, :]
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))

                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)

                edisp = s6 * t6 + s8 * rrij * t8

                de = -C6ij * edisp * 0.5
                # print(i + 1, j + 1, r2, r0ij, edisp, de)
                energies[i] += de
                if i != j:
                    energies[j] += de
    energy = np.sum(energies)
    return energy


def compute_bj_pairs_DIMER(
    params,
    r,
    r4r2_ls=r4r2.r4r2_vals_ls(),
):
    num, carts = r["Geometry_bohr"][:, 0], r["Geometry_bohr"][:, 1:]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]
    C6s = r["C6s"]

    energy = 0
    if len(params) == 3:
        s8, a1, a2 = params
        s6 = 1.0
    elif len(params) == 4:
        s6, s8, a1, a2 = params
    else:
        raise ValueError("params must be length 3 or 4")
    M_tot = len(carts)
    energies = np.zeros(M_tot)
    lattice_points = 1

    # for i in range(M_tot):
    for i in monAs:
        el1 = int(num[i])
        Q_A = (0.5 * el1**0.5 * r4r2_ls[el1 - 1]) ** 0.5

        for j in monBs:
            el2 = int(num[j])
            Q_B = (0.5 * el2**0.5 * r4r2_ls[el2 - 1]) ** 0.5
            if i == j:
                continue
            for k in range(lattice_points):
                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                C6ij = C6s[i, j]

                r1, r2 = carts[i, :], carts[j, :]
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))

                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)

                edisp = s6 * t6 + s8 * rrij * t8

                de = -C6ij * edisp * 0.5
                # print(i + 1, j + 1, r2, r0ij, edisp, de)
                energies[i] += de
                if i != j:
                    energies[j] += de
    energy = np.sum(energies)
    return energy * hartree_to_kcalmol


def compute_bj_dimer_f90(
    params,
    r,
    r4r2_ls=r4r2.r4r2_vals_ls(),
):
    num, coords = r["Geometry_bohr"][:, 0], r["Geometry_bohr"][:, 1:]
    charges = r["charges"]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]

    e_d = compute_bj_f90(
        num,
        coords,
        r["C6s"],
        params=params,
        r4r2_ls=r4r2_ls,
    )

    n1, p1 = num[monAs], coords[monAs, :]
    e_1 = compute_bj_f90(
        n1,
        p1,
        r["C6_A"],
        params=params,
        r4r2_ls=r4r2_ls,
    )

    n2, p2 = num[monBs], coords[monBs, :]
    e_2 = compute_bj_f90(
        n2,
        p2,
        r["C6_B"],
        params=params,
        r4r2_ls=r4r2_ls,
    )

    e_total = (e_d - (e_1 + e_2)) * hartree_to_kcalmol
    return e_total


def compute_bj_dimer_DFTD4(
    params,
    pos,
    carts,
    Ma,
    Mb,
    charges,
    mult_out=constants.conversion_factor("hartree", "kcal / mol"),
    s9=0.0,
) -> float:
    """
    computes dftd4 for dimer and each monomer and returns subtraction.
    """
    _, _, _, d = calc_dftd4_c6_c8_pairDisp2(
        pos,
        carts,
        charges[0],
        p=params,
        s9=s9,
    )

    mon_ca = carts[Ma]
    mon_pa = pos[Ma]
    _, _, _, A = calc_dftd4_c6_c8_pairDisp2(
        mon_pa,
        mon_ca,
        charges[1],
        p=params,
        s9=s9,
    )

    mon_cb = carts[Mb]
    mon_pb = pos[Mb]
    _, _, _, B = calc_dftd4_c6_c8_pairDisp2(
        mon_pb,
        mon_cb,
        charges[2],
        p=params,
        s9=s9,
    )

    AB = A + B
    print(f"disp dftd4 = {d} - ({AB}) = {d} - ({A} + {B})")
    disp = (d - (AB)) * mult_out
    return disp


def compute_bj_with_different_C6s(
    geom,
    ma,
    mb,
    charges,
    C6s,
    C6s_A,
    C6s_B,
    params,
    s9=0.0,
):
    C6s_monA_from_dimer = get_monomer_C6s_from_dimer(C6s, ma)
    C6s_monB_from_dimer = get_monomer_C6s_from_dimer(C6s, mb)
    geom[:, 1:] *= constants.conversion_factor("angstrom", "bohr")
    row = {
        "Geometry_bohr": geom,
        "C6s": C6s_dimer,
        "charges": charges,
        "monAs": ma,
        "monBs": mb,
        "C6_A": C6s_mA,
        "C6_B": C6s_mB,
    }
    d4_mons_individually = compute_bj_dimer_f90(
        params,
        row,
        r4r2_ls=r4r2_ls,
    )
    row = {
        "Geometry_bohr": geom,
        "C6s": C6s_dimer,
        "charges": charges,
        "monAs": ma,
        "monBs": mb,
        "C6_A": C6s_monA_from_dimer,
        "C6_B": C6s_monB_from_dimer,
    }
    d4_dimer = compute_bj_dimer_f90(
        params,
        row,
        r4r2_ls=r4r2_ls,
    )
    return d4_dimer, d4_mons_individually
