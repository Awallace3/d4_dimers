import pandas as pd
from glob import glob
import numpy as np
import json
from periodictable import elements
import pickle
import pprint as pp
# import d4
import dftd4

import argparse
import numpy as np
import pandas as pd
import scipy.optimize as opt
import time
from dftd4 import interface
import matplotlib.pyplot as plt


# Interaction energy = E_AB - E_A - E_B


def write_pickle(data, fname='data.pickle'):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fname='data.pickle'):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)

def generate_truncated_csv(path: str="master-regen.pkl"):
    """
    generate_truncated_csv generates truncated form of D3 paper results.
    """

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    df = read_pickle(path)
    for i in df.columns.values:
        print(i)
    # pp.pprint(df.columns.values)
    df = df[["DB", "Name", "HF INTERACTION ENERGY", "HF TOTAL ENERGY", "System #", "Geometry", "Benchmark"]]
    print(df[[ "HF INTERACTION ENERGY", "Benchmark" ]].head())
    df.to_csv("truncated.csv")
    return

def main():
    # create_data_csv()
    names = [
        # "ACHC",
        # "ACONF",
        # "C2H4_NT",
        "CH4_PAH",
        # "CO2_NPHAC",
        # "CO2_PAH",
        # "CYCONF",
        # "HBC6",
        # "NBC10ext",
        # "PCONF",
        # "S22by7",
        # "S66by10",
        # "SCONF",
        # "Water2510",
        # "X31by10",
    ]
    # run_sapt0_for_db(names)
    """
    >>> from dftd4.interface import DampingParam, DispersionModel
    >>> import numpy as np
    >>> numbers = np.array([1, 1, 6, 5, 1, 15, 8, 17, 13, 15, 5, 1, 9, 15, 1, 15])
    >>> positions = np.array([  # Coordinates in Bohr
    ...     [+2.79274810283778, +3.82998228828316, -2.79287054959216],
    ...     [-1.43447454186833, +0.43418729987882, +5.53854345129809],
    ...     [-3.26268343665218, -2.50644032426151, -1.56631149351046],
    ...     [+2.14548759959147, -0.88798018953965, -2.24592534506187],
    ...     [-4.30233097423181, -3.93631518670031, -0.48930754109119],
    ...     [+0.06107643564880, -3.82467931731366, -2.22333344469482],
    ...     [+0.41168550401858, +0.58105573172764, +5.56854609916143],
    ...     [+4.41363836635653, +3.92515871809283, +2.57961724984000],
    ...     [+1.33707758998700, +1.40194471661647, +1.97530004949523],
    ...     [+3.08342709834868, +1.72520024666801, -4.42666116106828],
    ...     [-3.02346932078505, +0.04438199934191, -0.27636197425010],
    ...     [+1.11508390868455, -0.97617412809198, +6.25462847718180],
    ...     [+0.61938955433011, +2.17903547389232, -6.21279842416963],
    ...     [-2.67491681346835, +3.00175899761859, +1.05038813614845],
    ...     [-4.13181080289514, -2.34226739863660, -3.44356159392859],
    ...     [+2.85007173009739, -2.64884892757600, +0.71010806424206],
    ... ])
    >>> model = DispersionModel(numbers, positions)
    >>> res = model.get_dispersion(DampingParam(method="scan"), grad=False)
    >>> res.get("energy")  # Results in atomic units
    -0.005328888532435093
    >>> res.update(**model.get_properties())  # also allows access to properties
    >>> res.get("c6 coefficients")[0, 0]
    1.5976689760849156
    >>> res.get("polarizibilities")
    array([ 1.97521745,  1.48512704,  7.33564674, 10.28920458,  1.99973802,
           22.85298573,  6.65877552, 15.39410319, 22.73119177, 22.86303028,
           14.56038118,  1.4815783 ,  3.91266859, 25.8236368 ,  1.93444627,
           23.02494331])
    """

    from dftd4.interface import DispersionModel, DampingParam
    import numpy as np
    disp = DispersionModel(
        numbers=np.array([16, 16, 16, 16, 16, 16, 16, 16]),
        positions=np.array([
            [-4.15128787379191, +1.71951973863958, -0.93066267097296],
            [-4.15128787379191, -1.71951973863958, +0.93066267097296],
            [-1.71951973863958, -4.15128787379191, -0.93066267097296],
            [+1.71951973863958, -4.15128787379191, +0.93066267097296],
            [+4.15128787379191, -1.71951973863958, -0.93066267097296],
            [+4.15128787379191, +1.71951973863958, +0.93066267097296],
            [+1.71951973863958, +4.15128787379191, -0.93066267097296],
            [-1.71951973863958, +4.15128787379191, +0.93066267097296],
        ]),
    )
    res = disp.get_pairwise_dispersion(DampingParam(method='tpss'))
    # print(res)
    plt.plot(range(10), range(10))
    plt.show()
    # plt.savefig('tmp.png')




if __name__ == "__main__":
    main()
