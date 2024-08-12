import pytest
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools
import pandas as pd
import sys, os
from dispersion import disp
import os
from pprint import pprint as pp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ""))
import src

from pathlib import Path

d3data_path = Path("~/gits/simple-dftd3/_build/app/s-dftd3").expanduser()
print(d3data_path)

data_pkl = Path(__file__).parent / "../plots/basis_study.pkl"

@pytest.fixture
def water1():
    return np.array(
    [[  1.0, 2.0, 1.83610616, 4.11355547,  5.43751469,  84.92790104,],
     [  1.0, 3.0, 1.83610615, 4.11355547,  5.43768865,  84.93061808,],
     [  1.0, 4.0, 5.75571609, 4.68973278, 10.41372269, 210.15459179,],
     [  1.0, 5.0, 6.51917415, 4.11355547,  5.43772497,  84.93118534,],
     [  1.0, 6.0, 7.35133411, 4.11355547,  5.43773537,  84.93134783,],
     [  2.0, 3.0, 2.90730364, 4.12394911,  3.09362389,  37.39680777,],
     [  2.0, 4.0, 4.46729175, 4.11355547,  5.43744483,  84.9268099 ,],
     [  2.0, 5.0, 5.35713867, 4.12394911,  3.09364703,  37.39708746,],
     [  2.0, 6.0, 5.92487738, 4.12394911,  3.09365365,  37.39716757,],
     [  3.0, 4.0, 5.34672146, 4.11355547,  5.43761879,  84.92952689,],
     [  3.0, 5.0, 6.15603982, 4.12394911,  3.09375785,  37.39842716,],
     [  3.0, 6.0, 6.96128977, 4.12394911,  3.09376448,  37.39850728,],
     [  4.0, 5.0, 1.83610617, 4.11355547,  5.43765511,  84.93009414,],
     [  4.0, 6.0, 1.83610615, 4.11355547,  5.43766551,  84.93025663,],
     [  5.0, 6.0, 2.90730364, 4.12394911,  3.09378762,  37.39878699,],
     [  1.0, 2.0, 1.83610616, 4.11355547, -5.43783367, -84.93288306,],
     [  1.0, 3.0, 1.83610615, 4.11355547, -5.43783367, -84.93288305,],
     [  2.0, 3.0, 2.90730364, 4.12394911, -3.09381064, -37.39906528,],
     [  1.0, 2.0, 1.83610617, 4.11355547, -5.43783367, -84.93288306,],
     [  1.0, 3.0, 1.83610615, 4.11355547, -5.43783367, -84.93288305,],
     [  2.0, 3.0, 2.90730364, 4.12394911, -3.09381064, -37.39906529,],]
)

def test_d3data_generation():    
    """
    For test to pass, require `git clone -b d3data git@github.com:awallace3/simple-dftd3.git`
    """
    df = pd.read_pickle(data_pkl)
    row = df.iloc[2500]
    target = np.array(row['D3Data'])
    print("TARGET")
    print(target)
    pos, carts = row['Geometry'][:, 0], row['Geometry'][:, 1:]
    d3data = src.dftd3.collect_bjm_d3data_dimer(pos, carts, row['monAs'], row['monBs'], ATM=False, s_dftd3_bin=d3data_path)
    print("D3Data")
    print(d3data)
    print(len(d3data), len(target))
    print('rs')
    np.testing.assert_allclose(d3data[:, 2], target[:, 2], atol=1e-4)
    assert np.allclose(d3data[:, 2], target[:, 2], atol=1e-4)
    # Don't need to compare r0s because jeff.py calculates r0 from c6s and c8s before use
    # print('r0s')
    # np.testing.assert_allclose(d3data[:, 3], target[:, 3], atol=1e-4)
    print('c6s')
    np.testing.assert_allclose(d3data[:, 4], target[:, 4], atol=1e-4)
    assert np.allclose(d3data[:, 4], target[:, 4], atol=1e-4)
    print('c8s')
    np.testing.assert_allclose(d3data[:, 5], target[:, 5], atol=1e-4)
    assert np.allclose(d3data[:, 5], target[:, 5], atol=1e-4)

    test_bj = src.jeff.compute_BJ_CPP(np.array([0.713190, 0.079541, 3.627854]), d3data)
    actual_bj = src.jeff.compute_BJ_CPP(np.array([0.713190, 0.079541, 3.627854]), target)
    print(test_bj, actual_bj)
    assert np.allclose(test_bj, actual_bj, atol=1e-8)
    return

if __name__ == "__main__":
    test_d3data_generation()
