import pytest
import numpy as np
import qcelemental as qcel
from qm_tools_aw import tools as tools
import pandas as pd
from psi4.driver.wrapper_database import database
import psi4
import sys, os
from dispersion import disp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ""))
import src

ang_to_bohr_qcel = qcel.constants.conversion_factor("angstrom", "bohr")
ang_to_bohr = src.constants.Constants().g_aatoau()
hartree_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

# You will need to build https://github.com/Awallace3/dftd4 for pytest to pass
dftd4_bin = "/theoryfs2/ds/amwalla3/.local/bin/dftd4"
data_pkl = "/theoryfs2/ds/amwalla3/projects/d4_corrections/tests/data/sr.pkl"

# y_p: -0.3581934534333338, y: 1.3651143133442858
