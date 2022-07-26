import pytest
from src.setup import calc_dftd4_props, compute_bj_self
import numpy as np
import math
from src.r4r2 import r4r2_vals
import json


def compute_bj_self_test_setup():
    params = [1.0000, 0.9, 0.4, 5.0]
    pos = np.array(
        [
            7.0,
            6.0,
            7.0,
            6.0,
            6.0,
            6.0,
            7.0,
            6.0,
            7.0,
            1.0,
            7.0,
            1.0,
            1.0,
            1.0,
            1.0,
            7.0,
            6.0,
            7.0,
            6.0,
            6.0,
            6.0,
            1.0,
            8.0,
            7.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    carts = np.array(
        [
            [-2.143860e-01, -1.993021e00, 1.580000e-04],
            [-1.483458e00, -1.537649e00, 3.900000e-05],
            [-1.896987e00, -2.697650e-01, -1.260000e-04],
            [-8.590920e-01, 5.833320e-01, -1.300000e-04],
            [5.023510e-01, 2.582240e-01, 2.500000e-05],
            [8.189870e-01, -1.120415e00, 1.690000e-04],
            [1.293025e00, 1.395362e00, -1.800000e-05],
            [4.308840e-01, 2.385528e00, -1.910000e-04],
            [-8.846300e-01, 1.960299e00, -2.680000e-04],
            [-2.247335e00, -2.317004e00, 5.500000e-05],
            [2.078153e00, -1.590279e00, 3.130000e-04],
            [2.839337e00, -9.285350e-01, 3.540000e-04],
            [2.257625e00, -2.599226e00, 5.070000e-04],
            [6.864280e-01, 3.440469e00, -2.720000e-04],
            [-1.715093e00, 2.535836e00, -4.080000e-04],
            [9.890610e-01, 7.972580e-01, 3.400058e00],
            [1.217664e00, -5.930370e-01, 3.400004e00],
            [1.323910e-01, -1.408972e00, 3.399965e00],
            [-1.110962e00, -9.038440e-01, 3.399971e00],
            [-1.352449e00, 5.216270e-01, 3.400023e00],
            [-2.635560e-01, 1.333781e00, 3.400063e00],
            [1.815200e00, 1.380954e00, 3.400092e00],
            [2.388816e00, -9.949650e-01, 3.399991e00],
            [-2.130425e00, -1.768706e00, 3.399929e00],
            [-1.956075e00, -2.796574e00, 3.399871e00],
            [-3.078136e00, -1.425139e00, 3.399928e00],
            [-2.359817e00, 9.292250e-01, 3.400032e00],
            [-3.321890e-01, 2.420853e00, 3.400106e00],
        ]
    )
    Ma = 15
    Mb = 13
    C6s = calc_dftd4_props(pos, carts)
    return params, pos, carts, Ma, Mb, C6s


def read_dftd4_vals():
    with open("vals.json") as f:
        d = json.load(f)
    for k, v in d.items():
        d[k] = np.array(v)
    return d


def compute_bj_self_pieces():
    params, pos, carts, Ma, Mb, C6s = compute_bj_self_test_setup()
    rrijs, r2s, t6s, t8s, edisps, des = [], [], [], [], [], []

    electron_mass = 9.1093837015e-31
    c = 299792458
    fine_structure_constant = 7.2973525693e-3
    hbar = 6.62607015e-34 / (2 * math.pi)
    bohr = hbar / (electron_mass * c * fine_structure_constant)
    autoaa = bohr * 1e10
    aatoau = 1 / autoaa

    energy = 0
    s6, s8, a1, a2 = params
    C8s = np.zeros(np.shape(C6s))
    N_tot = len(carts)
    energies = np.zeros(Ma + Mb)
    M_tot = len(carts)
    lattice_points = 1

    cs = aatoau * np.array(carts, copy=True)
    for i in range(M_tot):
        el1 = int(pos[i])
        el1_r4r2 = r4r2_vals(el1)
        Q_A = np.sqrt(np.sqrt(el1) * el1_r4r2)

        for j in range(i):
            if i == j:
                continue
            for k in range(lattice_points):
                el2 = int(pos[j])
                el2_r4r2 = r4r2_vals(el2)
                Q_B = np.sqrt(np.sqrt(el2) * el2_r4r2)

                rrij = 3 * Q_A * Q_B
                r0ij = a1 * np.sqrt(rrij) + a2
                C6 = C6s[i, j]
                r1, r2 = cs[i, :], cs[j, :]
                r2 = np.subtract(r1, r2)
                r2 = np.sum(np.multiply(r2, r2))
                # R value is not a match since dftd4 converts input carts
                t6 = 1 / (r2**3 + r0ij**6)
                t8 = 1 / (r2**4 + r0ij**8)
                edisp = s6 * t6 + s8 * rrij * t8
                edisp = s6 * t6 + s8 * rrij * t8

                de = -C6 * edisp * 0.5
                energies[i] += de
                if i != j:
                    energies[j] += de
                rrijs.append(rrij)
                r2s.append(r2)
                t6s.append(t6)
                t8s.append(t8)
                edisps.append(edisp)
                des.append(de)

    # print(len(rrijs))
    # print(M_tot * (M_tot - 1) / 2)
    energy = np.sum(energies)
    return rrijs, r2s, t6s, t8s, edisps, des, energies, energy


rrijs, r2s, t6s, t8s, edisps, des, energies, energy = compute_bj_self_pieces()
d = read_dftd4_vals()
rrijs_t, r2s_t, t6s_t, t8s_t, edisps_t, des_t, energies_t = (
    d["rrijs"],
    d["r2s"],
    d["t6s"],
    d["t8s"],
    d["edisps"],
    d["des"],
    d["energy"],
)
# @pytest.mark.parametrize("rrijs", rrijs)
# @pytest.mark.parametrize("rrijs_t", rrijs_t)
def test_bj_rrij():
    """
    dispersion2
    _t are reported up to 6 decimal places
    """
    t = np.abs(np.subtract(rrijs, rrijs_t))
    assert np.all(t < 1e-6)


def test_bj_r2s():
    """
    dispersion2
    _t are reported up to 6 decimal places
    """
    t = np.abs(np.subtract(r2s, r2s_t))
    assert np.all(t < 1e-5)


def test_bj_t6s():
    """
    dispersion2
    _t are reported up to 16 decimal places
    """
    t = np.abs(np.subtract(t6s, t6s_t))
    assert np.all(t < 1e-16)


def test_bj_t8s():
    """
    dispersion2
    _t are reported up to 16 decimal places
    """
    t = np.abs(np.subtract(t8s, t8s_t))
    assert np.all(t < 1e-16)


def test_bj_edisps():
    """
    dispersion2
    _t are reported up to 16 decimal places
    """
    t = np.abs(np.subtract(edisps, edisps_t))
    assert np.all(t < 1e-16)


def test_bj_des():
    """
    dispersion2
    _t are reported up to 16 decimal places
    """
    t = np.abs(np.subtract(des, des_t))
    assert np.all(t < 1e-16)


def test_bj_energies():
    """
    dispersion2
    _t are reported up to 16 decimal places
    """
    t = np.abs(np.subtract(energies, energies_t))
    assert np.all(t < 1e-16)


def test_bj_energies():
    """
    dispersion2
    _t are reported up to 16 decimal places
    """
    e = np.sum(energies)
    e_t = np.sum(energies_t)
    t = np.abs(e - e_t)
    assert t < 1e-16


def compute_parameters():
    # rrijs, r2s, t6s, t8s, edisps, des, energies, energy = compute_bj_self_pieces()
    # d = read_dftd4_vals()
    # rrijs_t, r2s_t, t6s_t, t8s_t, edisps_t, des_t = (
    #     d["rrijs"],
    #     d["r2s"],
    #     d["t6s"],
    #     d["t8s"],
    #     d["edisps"],
    #     d["des"],
    # )
    # test_bj_rrij(rrijs, rrijs_t)
    # print(edisps[16], edisps_t[16], np.subtract(edisps[16], edisps_t[16]))
    t = np.zeros((len(r2s), 3))
    t[:, 0] = r2s
    t[:, 1] = r2s_t
    t[:, 2] = t[:, 0] - t[:, 1]
    for i in t:
        print(i)

    return


def main():
    """
    docstring
    """
    compute_parameters()
    return


if __name__ == "__main__":
    main()
