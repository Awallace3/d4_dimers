import numpy as np
import qcelemental as qcel
import subprocess
import os
import json
from . import r4r2
from qm_tools_aw import tools
from dispersion import disp
from pprint import pprint as pp

hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")


def compute_psi4_d4(geom, Ma, Mb, memory: str = "4 GB", basis="jun-cc-pvdz"):
    ma, mb = split_dimer(geom, Ma, Mb)
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)
    geom = "0 1\n%s--\n0 1\n%s" % (ma, mb)
    print(geom)
    psi4.geometry(geom)
    psi4.set_memory(memory)
    psi4.set_options(
        {
            "basis": basis,
            "freeze_core": "true",
            "guess": "sad",
            "scf_type": "df",
        }
    )
    v = psi4.energy("hf-d4", bsse_type="cp")
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


def calc_dftd4_c6_c8_pairDisp2(
    atom_numbers: np.array,
    carts: np.array,  # angstroms
    charges: np.array,
    input_xyz: str = "dat.xyz",
    dftd4_bin: str = "/theoryfs2/ds/amwalla3/.local/bin/dftd4",
    p: [] = [1.0, 1.61679827, 0.44959224, 3.35743605],
    s9=0.0,
    C6s_ATM=False,
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
    v = subprocess.call(
        args=args,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    assert v == 0
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
    os.remove(".EDISP")
    if C6s_ATM:
        with open("C_n_ATM.json") as f:
            cs = json.load(f)
        C6s_ATM = np.array(cs["c6_ATM"], dtype=np.float64)
        os.remove("C_n_ATM.json")
        return C6s, C8s, pairs, e, C6s_ATM
    else:
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
                energies[i] += de
                if i != j:
                    energies[j] += de
    energy = np.sum(energies)
    return energy


def triple_scale(ii, jj, kk) -> float:
    """
    triple_scale distribute a triple energy to atomwise energies
    """
    if ii == jj:
        if ii == kk:
            triple = 1 / 6
        else:
            triple = 0.5
    else:
        if ii != kk and jj != kk:
            triple = 1
        else:
            triple = 0.5
    return triple


def compute_bj_f90_ATM(
    pos: np.array,
    carts: np.array,
    C6s_ATM: np.array,
    params: [] = [1.61679827, 0.44959224, 3.35743605, 1.0],
    # [s6, s8, a1, a2, s9]
    r4r2_ls=r4r2.r4r2_vals_ls(),
    alp=16.0,
    ATM_only=False,
    C6s: np.array = None,
) -> float:
    """
    compute_bj_f90 computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    if ATM_only:
        e_two_body_disp = 0
    else:
        if len(params) >= 4:
            p2b = params[:4]
        else:
            p2b = params[:3]

        e_two_body_disp = compute_bj_f90(
            pos,
            carts,
            C6s,
            params=p2b,
            r4r2_ls=r4r2_ls,
        )
    energy = 0
    if len(params) == 3:
        s8, a1, a2 = params
        s6, s9 = 1.0, 1.0
    else:
        s6, s8, a1, a2, s9 = params
    M_tot = len(carts)
    energies = np.zeros((M_tot, M_tot))
    lattice_points = 1
    e_ATM = 0
    for i in range(M_tot):
        el1 = int(pos[i])
        Q_A = (0.5 * el1**0.5 * r4r2_ls[el1 - 1]) ** 0.5
        for j in range(i + 1):
            el2 = int(pos[j])
            Q_B = (0.5 * el2**0.5 * r4r2_ls[el2 - 1]) ** 0.5
            c6ij = C6s_ATM[i, j]
            r0ij = a1 * np.sqrt(3 * Q_A * Q_B) + a2
            ri, rj = carts[i, :], carts[j, :]
            r2ij = np.subtract(ri, rj)
            r2ij = np.sum(np.multiply(r2ij, r2ij))
            if np.all(r2ij < 1e-8):
                continue
            for k in range(j + 1):
                el3 = int(pos[k])
                Q_C = (0.5 * el3**0.5 * r4r2_ls[el3 - 1]) ** 0.5
                c6ik = C6s_ATM[i, k]
                c6jk = C6s_ATM[j, k]
                c9 = -s9 * np.sqrt(np.abs(c6ij * c6ik * c6jk))
                r0ik = a1 * np.sqrt(3 * Q_C * Q_A) + a2
                r0jk = a1 * np.sqrt(3 * Q_C * Q_B) + a2
                r0 = r0ij * r0ik * r0jk
                triple = triple_scale(i, j, k)
                for ktr in range(lattice_points):
                    rk = carts[k, :]
                    r2ik = np.subtract(ri, rk)
                    r2ik = np.sum(np.multiply(r2ik, r2ik))
                    if np.all(r2ik < 1e-8):
                        continue
                    r2jk = np.subtract(rj, rk)
                    r2jk = np.sum(np.multiply(r2jk, r2jk))
                    if np.all(r2jk < 1e-8):
                        continue
                    r2 = r2ij * r2ik * r2jk
                    r1 = np.sqrt(r2)
                    r3 = r2 * r1
                    r5 = r3 * r2

                    fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** (alp / 3.0))

                    ang = (
                        0.375
                        * (r2ij + r2jk - r2ik)
                        * (r2ij - r2jk + r2ik)
                        * (-r2ij + r2jk + r2ik)
                        / r5
                        + 1.0 / r3
                    )

                    rr = ang * fdmp

                    dE = rr * c9 * triple / 6
                    e_ATM -= dE
                    energies[j, i] -= dE
                    energies[k, i] -= dE
                    energies[i, j] -= dE
                    energies[k, j] -= dE
                    energies[i, k] -= dE
                    energies[j, k] -= dE
    energy = np.sum(energies)
    if not ATM_only:
        energy += e_two_body_disp
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


def compute_disp_2B(
    params,
    r,
    params_ATM=None,
):
    num, coords = (
        np.array(r["Geometry_bohr"][:, 0], dtype=np.int32),
        r["Geometry_bohr"][:, 1:],
    )
    charges = r["charges"]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]

    e_d = disp.disp_2B(
        num,
        coords,
        r["C6s"],
        params,
    )
    n1, p1 = num[monAs], coords[monAs, :]
    e_1 = disp.disp_2B(
        n1,
        p1,
        r["C6_A"],
        params,
    )

    n2, p2 = num[monBs], coords[monBs, :]
    e_2 = disp.disp_2B(
        n2,
        p2,
        r["C6_B"],
        params,
    )

    e_total = (e_d - (e_1 + e_2)) * hartree_to_kcalmol
    return e_total


def compute_disp_2B_dimer(
    params,
    r,
):
    pos, carts = (
        np.array(r["Geometry_bohr"][:, 0], dtype=np.int32),
        r["Geometry_bohr"][:, 1:],
    )
    charges = r["charges"]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]
    pA, cA = pos[monAs], carts[monAs, :]
    pB, cB = pos[monBs], carts[monBs, :]
    e_total = disp.disp_2B_dimer(
        pos, carts, r["C6s"], pA, cA, r["C6_A"], pB, cB, r["C6_B"], params
    )
    return e_total * hartree_to_kcalmol


def compute_disp_ATM_CHG_dimer(
    params,
    r,
    mult_out=hartree_to_kcalmol,
):
    pos, carts = (
        np.array(r["Geometry_bohr"][:, 0], dtype=np.int32),
        r["Geometry_bohr"][:, 1:],
    )
    charges = r["charges"]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]
    pA, cA = pos[monAs], carts[monAs, :]
    pB, cB = pos[monBs], carts[monBs, :]
    e_total = disp.disp_ATM_CHG_dimer(
        pos, carts, r["C6_ATM"], pA, cA, r["C6_ATM_A"], pB, cB, r["C6_ATM_B"], params
    )
    return e_total * mult_out


def compute_disp_2B_BJ_ATM_CHG_dimer(
    r,
    params_2B,
    params_ATM,
    mult_out=hartree_to_kcalmol,
):
    pos, carts = (
        np.array(r["Geometry_bohr"][:, 0], dtype=np.int32),
        r["Geometry_bohr"][:, 1:],
    )
    charges = r["charges"]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]
    pA, cA = pos[monAs], carts[monAs, :]
    pB, cB = pos[monBs], carts[monBs, :]
    e_total = disp.disp_2B_BJ_ATM_CHG(
        pos,
        carts,
        r["C6s"],
        r["C6_ATM"],
        pA,
        cA,
        r["C6_A"],
        r["C6_ATM_A"],
        pB,
        cB,
        r["C6_B"],
        r["C6_ATM_B"],
        params_2B,
        params_ATM,
    )
    return e_total * mult_out


def compute_disp_2B_BJ_ATM_TT_dimer(
    r,
    params_2B,
    params_ATM,
    mult_out=hartree_to_kcalmol,
):
    pos, carts = (
        np.array(r["Geometry_bohr"][:, 0], dtype=np.int32),
        r["Geometry_bohr"][:, 1:],
    )
    charges = r["charges"]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]
    pA, cA = pos[monAs], carts[monAs, :]
    pB, cB = pos[monBs], carts[monBs, :]
    e_total = disp.disp_2B_BJ_ATM_TT(
        pos,
        carts,
        r["C6s"],
        r["C6_ATM"],
        pA,
        cA,
        r["C6_A"],
        r["C6_ATM_A"],
        pB,
        cB,
        r["C6_B"],
        r["C6_ATM_B"],
        params_2B,
        params_ATM,
    )
    # e_total = disp.disp_ATM_TT_dimer(
    #     pos,
    #     carts,
    #     r["C6_ATM"],
    #     pA,
    #     cA,
    #     r["C6_ATM_A"],
    #     pB,
    #     cB,
    #     r["C6_ATM_B"],
    #     params_ATM,
    # )
    return e_total * mult_out


def compute_disp_2B_BJ_ATM_SR_dimer(
    r,
    params_2B,
    params_ATM,
    mult_out=hartree_to_kcalmol,
    SR_func=disp.disp_SR_1,
):
    pos, carts = (
        np.array(r["Geometry_bohr"][:, 0], dtype=np.int32),
        r["Geometry_bohr"][:, 1:],
    )
    charges = r["charges"]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]
    pA, cA = pos[monAs], carts[monAs, :]
    pB, cB = pos[monBs], carts[monBs, :]
    e_d = SR_func(
        pos,
        carts,
        r["C6_ATM"],
        params_ATM,
    )
    e_1 = SR_func(
        pA,
        cA,
        r["C6_ATM_A"],
        params_ATM,
    )
    e_2 = SR_func(
        pB,
        cB,
        r["C6_ATM_B"],
        params_ATM,
    )
    e_total = e_d - (e_1 + e_2)
    e_total *= mult_out
    return e_total


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


def compute_bj_dimer_f90_ATM(
    params,
    r,
    r4r2_ls=r4r2.r4r2_vals_ls(),
):
    num, coords = r["Geometry_bohr"][:, 0], r["Geometry_bohr"][:, 1:]
    charges = r["charges"]
    monAs, monBs = r["monAs"], r["monBs"]
    charges = r["charges"]

    e_d = compute_bj_f90_ATM(
        num,
        coords,
        r["C6_ATM"],
        C6s=r["C6s"],
        params=params,
        r4r2_ls=r4r2_ls,
    )

    n1, p1 = num[monAs], coords[monAs, :]
    e_1 = compute_bj_f90_ATM(
        n1,
        p1,
        r["C6_ATM_A"],
        C6s=r["C6_A"],
        params=params,
        r4r2_ls=r4r2_ls,
    )

    n2, p2 = num[monBs], coords[monBs, :]
    e_2 = compute_bj_f90_ATM(
        n2,
        p2,
        r["C6_ATM_B"],
        C6s=r["C6_B"],
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
    mult_out=hartree_to_kcalmol,
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


def compute_bj_dimer_DFTD4_ATM(
    params,
    pos,
    carts,
    Ma,
    Mb,
    charges,
    mult_out=hartree_to_kcalmol,
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
    r4r2_ls=r4r2.r4r2_vals_ls(),
):
    C6s_monA_from_dimer = get_monomer_C6s_from_dimer(C6s, ma)
    C6s_monB_from_dimer = get_monomer_C6s_from_dimer(C6s, mb)
    row = {
        "Geometry_bohr": geom,
        "C6s": C6s,
        "charges": charges,
        "monAs": ma,
        "monBs": mb,
        "C6_A": C6s_A,
        "C6_B": C6s_B,
    }
    d4_mons_individually = compute_bj_dimer_f90(
        params,
        row,
        r4r2_ls=r4r2_ls,
    )
    row = {
        "Geometry_bohr": geom,
        "C6s": C6s,
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


def compute_bj_f90_NO_DAMPING(
    pos: np.array,
    carts: np.array,
    C6s: np.array,
    r4r2_ls=r4r2.r4r2_vals_ls(),
) -> float:
    """
    compute_bj_f90 computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    energy = 0
    M_tot = len(carts)
    energies = np.zeros(M_tot)
    lattice_points = 1
    aatoau = Constants().g_aatoau()
    cs = aatoau * np.array(carts, copy=True)
    for i in range(M_tot):
        el1 = int(pos[i])
        Q_A = (0.5 * el1**0.5 * r4r2_ls[el1 - 1]) ** 0.5
        for j in range(i):
            if i == j:
                continue
            for k in range(lattice_points):
                el2 = int(pos[j])
                Q_B = (0.5 * el2**0.5 * r4r2_ls[el2 - 1]) ** 0.5

                C6 = C6s[i, j]
                C8 = 3 * C6 * np.sqrt(Q_A * Q_B)
                r1, r2 = cs[i, :], cs[j, :]
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))
                r2 = np.sqrt(r2)
                print(i, j, int(pos[i]), int(pos[j]), C6, C8, r2, Q_A, Q_B)
                R_6 = r2**6
                R_8 = r2**8

                de = C6 / R_6 + C8 / R_8
                energies[i] += de
                if i != j:
                    energies[j] += de
    energy = -np.sum(energies)
    return energy
