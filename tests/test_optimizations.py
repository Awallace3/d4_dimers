import pytest
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools as tools
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ""))
import src

ang_to_bohr_qcel = qcel.constants.conversion_factor("angstrom", "bohr")
ang_to_bohr = src.constants.Constants().g_aatoau()
hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

dftd4_bin = "/theoryfs2/ds/amwalla3/.local/bin/dftd4"
data_pkl = "/theoryfs2/ds/amwalla3/projects/d4_corrections/tests/data/test.pkl"
data_pkl_g = "/theoryfs2/ds/amwalla3/projects/d4_corrections/tests/data/g_t1.pkl"


@pytest.mark.parametrize(
    "params, HF_key, target_rmse",
    [
        (
            np.array([1.0, 1.61679827, 0.44959224, 3.35743605, 1.0], dtype=np.float64),
            "HF_qz",
            0.47519209,
        ),
        (
            np.array([1.0, 1.04488291, 0.41144968, 3.14951807, 0.0], dtype=np.float64),
            "HF_qz",
            0.45367290,
        ),
        (
            [
                np.array(
                    [1.0, 1.04488291, 0.41144968, 3.14951807, 0.0], dtype=np.float64
                ),
                np.array(
                    [1.0, 1.04488291, 0.41144968, 3.14951807, 0.0], dtype=np.float64
                ),
            ],
            "HF_qz",
            0.45367290,
        ),
        (
            np.array(
                [1.0, 1.61679827, 0.44959224, 3.35743605, 0.44959224, 3.35743605, 1.0],
                dtype=np.float64,
            ),
            "HF_qz",
            0.47519209,
        ),
        (
            np.array(
                [1.61679827, 0.44959224, 3.35743605, 0.44959224, 3.35743605, 1.0],
                dtype=np.float64,
            ),
            "HF_qz",
            0.47519209,
        ),
        (
            src.paramsTable.get_params("HF_ATM_OPT_START"),
            "HF_qz",
            0.47519209,
        ),
        (
            src.paramsTable.get_params("HF_OPT"),
            "HF_qz",
            0.5619978691447731,
        ),
    ],
)
def test_compute_stats_DISP(params, HF_key, target_rmse):
    df = pd.read_pickle(data_pkl_g)
    print(params)
    mae, rmse, max_e, mad, mean_diff = src.optimization.compute_int_energy_stats_DISP(
        params,
        df,
        HF_key,
        print_results=True,
    )
    assert abs(rmse - target_rmse) < 1e-7


# @pytest.mark.parametrize(
#     "params, HF_key, target_rmse, ATM_ON",
#     [
#         ([1.61679827, 0.44959224, 3.35743605], "HF_qz", 0.47519209, True),
#         ([1.04488291, 0.41144968, 3.14951807], "HF_qz", 0.45367290, False),
#     ],
# )
# def test_compute_stats(params, HF_key, target_rmse, ATM_ON):
#     df = pd.read_pickle(data_pkl_g)
#     mae, rmse, max_e, mad, mean_diff = src.optimization.compute_int_energy_stats(
#         params,
#         df,
#         HF_key,
#         print_results=True,
#         ATM=ATM_ON,
#     )
#     assert abs(rmse - target_rmse) < 1e-7
