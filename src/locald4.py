import numpy as np
from periodictable import elements
import qcelemental as qcel
import subprocess
import os
import json
from . import r4r2
from .constants import Constants


def write_xyz_from_np(
    atom_numbers, carts, outfile="dat.xyz", charges=[0, 1]
) -> None:
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
    carts: np.array,
    charges: np.array,
    input_xyz: str = "dat.xyz",
    dftd4_bin: str = "/theoryfs2/ds/amwalla3/.local/bin/dftd4",
    p: [] = [1.0, 1.61679827, 0.44959224, 3.35743605],
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
        "0.0",
        "-c",
        str(charges[0]),
        "--pair-resolved",
    ]
    print(" ".join(args))
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
    with open(".EDISP", 'r') as f:
        e = float(f.read())
    # os.remove(input_xyz)
    # os.remove("C_n.json")
    # os.remove("pairs.json")
    # subprocess.call(["cp",".EDISP", ".EDISP.d4fork"])
    return C6s, C8s, pairs, e


def compute_dispersion(num, coords, C6s, print_row=False, conv=False):
    C8s = np.copy(C6s)
    # print(f"{C6s = }")
    r4r2_ls = r4r2.r4r2_vals_ls()
    energy = 0
    for i in range(len(coords)):
        a = int(num[i])
        # print(r4r2_ls[a-1])
        Q_A = 0.5 * np.sqrt(a) * r4r2_ls[a - 1]
        for j in range(i + 1):
            b = int(num[j])
            Q_B = 0.5 * np.sqrt(b) * r4r2_ls[b - 1]
            C6 = C6s[i, j]
            C8 = 3 * C6 * np.sqrt(Q_A * Q_B)
            # print(C6, C8)
            # gathering C8s for comparison
            C8s[i, j] = C8
            C8s[j, i] = C8
            if i == j:
                continue
            r1, r2 = coords[i, :], coords[j, :]
            R_AB = np.sqrt(np.sum(np.square(np.subtract(r1, r2))))
            # use HF S8 value
            energy += C6 / R_AB**6 + C8 / R_AB**8
            energy += C6 / R_AB**6 + C8 / R_AB**8
            if print_row:
                print(i, j, a, b, C6, C8, R_AB, Q_A, Q_B)
    if conv:
        energy *= qcel.constants.conversion_factor("hartree", "kcal / mol")
    return -energy, C6s, C8s


def energy_tests():
    # Induction = (A<-B) + Induction (A->B) + delta HF, r(2)
    ind = [
        -1.12082991,  # Induction (A<-B)
        -3.47204886,  # Induction (A->B)
        -3.40555631,  # delta HF,r (2)
    ]
    print(sum(ind))

    disp = [
        -11.44771510,
        -15.55076869,
        3.43341555,
        4.66400942,
    ]
    print(sum(disp))

    # Dispersion = Disp2, r + Est. Exch-Disp2,r
    disp = [
        -11.44771510,  # Disp2,r
        3.43341555,  # Est. Exch-Disp2,r
    ]
    print(sum(disp))  # =  -8.01429955

    t = [-28.17807955, 23.58520078, -3.40555631]
    print(sum(t))
    # if Exch-Ind2,r under S^2 then should print
    return


"""
def dftd4_api_energy(num, coords, param_method="HF"):
    model = DispersionModel(num, coords)
    conv = qcel.constants.conversion_factor("hartree", "kcal/mol")
    res = model.get_dispersion(DampingParam(method=param_method), grad=False)
    edisp = res.get("energy")  # Results in atomic units
    return edisp


def dftd4_results():
    coords = np.array(
        [
            [8, 0.0370878820, 0.0000000000, -0.0551122700],
            [1, -0.7006692913, 0.5931409324, 0.1638729949],
            [1, 0.1120576564, -0.5931409324, 0.7107987731],
            [8, -0.0116916226, -0.0550639593, 2.2876732163],
            [1, 0.7884890539, 0.1683899814, 2.7914941964],
            [1, -0.6029345083, 0.7055150610, 2.4141950038],
        ]
    )
    num = coords[:, 0]
    ang_to_bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    coords = coords[:, 1:] * ang_to_bohr

    model = DispersionModel(num, coords)
    conv = qcel.constants.conversion_factor("hartree", "kcal/mol")

    res = model.get_dispersion(DampingParam(method="pbe0"), grad=False)
    edisp = res.get("energy")  # Results in atomic units
    print("pbe0 params:", edisp * conv, "kcal/mol")
    pp(get_damping_param("pbe0"))

    res = model.get_dispersion(DampingParam(method="HF"), grad=False)
    edisp = res.get("energy")  # Results in atomic units
    print("\nhf params:", edisp * conv, "kcal/mol")
    pp(get_damping_param("hf"))
    return
"""

def read_EDISP() -> None:
    """
    read_EDISP returns dftd4 .EDISP value
    """
    with open(".EDISP", 'r') as f:
        return float(f.read())


"""
def dftd4_api(pos, carts) -> ([], [], [], []):
    model = DispersionModel(pos, carts)
    res = model.get_properties()
    C6s = res.get("c6 coefficients")
    a = res.get("polarizibilities")
    cn = res.get("coordination numbers")
    pcharges = res.get("partial charges")
    return C6s, a, cn, pcharges
"""


def compute_bj_f90(
    pos: np.array,
    carts: np.array,
    C6s: np.array,
    params: [] = [1.61679827, 0.44959224, 3.35743605]
) -> float:
    """
    compute_bj_f90 computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    energy = 0
    s8, a1, a2 = params
    s6 = 1.0
    M_tot = len(carts)
    energies = np.zeros(M_tot)
    lattice_points = 1
    aatoau = Constants().g_aatoau()
    # cs = aatoau * np.array(carts, copy=True)
    cs = carts
    for i in range(M_tot):
        el1 = int(pos[i])
        el1_r4r2 = r4r2.r4r2_vals(el1)
        # TODO: add 0.5 in sqrt
        # Q_A = np.sqrt(np.sqrt(el1) * el1_r4r2)
        Q_A = np.sqrt(0.5 * np.sqrt(el1) * el1_r4r2)

        for j in range(i):
            if i == j:
                continue
            for k in range(lattice_points):
                el2 = int(pos[j])
                el2_r4r2 = r4r2.r4r2_vals(el2)
                Q_B = np.sqrt(0.5 * np.sqrt(el2) * el2_r4r2)

                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                C6 = C6s[i, j]

                r1, r2 = cs[i, :], cs[j, :]
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))
                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)
                edisp = s6 * t6 + s8 * rrij * t8

                de = -C6 * edisp * 0.5
                # de = -C6 * edisp
                energies[i] += de
                if i != j:
                    energies[j] += de
    energy = np.sum(energies)
    return energy


def compute_bj_f90_eff(
    pos: np.array,
    carts: np.array,
    C6s: np.array,
    params: [] = [1.61679827, 0.44959224, 3.35743605]
) -> float:
    """
    compute_bj_f90 computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    energy = 0
    s8, a1, a2 = params
    s6 = 1.0
    M_tot = len(carts)
    energies = np.zeros(M_tot)
    lattice_points = 1
    cs = carts
    for i in range(M_tot):
        el1 = int(pos[i])
        Q_A = r4r2.r4r2_vals_eff(el1)

        for j in range(i + 1):
            el2 = int(pos[j])
            Q_B = r4r2.r4r2_vals_eff(el2)
            if i == j:
                continue
            for k in range(lattice_points):

                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                C6ij = C6s[i, j]

                r1, r2 = cs[i, :], cs[j, :]
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


def compute_bj_f90_simplified(
    pos: np.array,
    carts: np.array,
    C6s: np.array,
    params: [] = [1.61679827, 0.44959224, 3.35743605]
) -> float:
    """
    compute_bj_f90 computes energy from C6s, cartesian coordinates, and dimer sizes.
    """
    energy = 0
    s8, a1, a2 = params
    s6 = 1.0
    M_tot = len(carts)
    energies = np.zeros(M_tot)
    lattice_points = 1
    cs = carts
    r4r2_ls = r4r2.r4r2_vals_ls()

    for i in range(M_tot):
        el1 = int(pos[i])
        Q_A = (0.5 * el1 ** 0.5 * r4r2_ls[el1 - 1]) ** 0.5

        for j in range(i + 1):
            el2 = int(pos[j])
            Q_B = (0.5 * el2 ** 0.5 * r4r2_ls[el2 - 1]) ** 0.5
            if i == j:
                continue
            for k in range(lattice_points):

                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                C6ij = C6s[i, j]

                r1, r2 = cs[i, :], cs[j, :]
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
