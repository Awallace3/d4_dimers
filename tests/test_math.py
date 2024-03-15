import pytest
import numpy as np
from dispersion import disp

def test_factorial():
    assert disp.factorial(5) == 120
    assert np.math.factorial(5) == 120

def test_f_TT_summation():
    b_ij = 1.5
    R_ij = 2.5
    n = 6
    val = 0
    R_b_ij = R_ij*b_ij
    for i in range(1,n+1):
        val += (R_b_ij**i)/np.math.factorial(i)
    assert disp.f_n_TT_summation(b_ij * R_ij, n) == val

def test_f_n_TT():
    b_ij = 1.5
    R_ij = 2.5
    n = 6
    val = 0
    R_b_ij = R_ij*b_ij
    for i in range(1,n+1):
        val += (R_b_ij**i)/np.math.factorial(i)
    final = 1 - np.exp(-R_b_ij) * val
    assert abs(disp.f_n_TT(b_ij, R_ij, n) - final) < 1e-8




