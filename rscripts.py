import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qm_tools_aw


def main():
    """
    random one time functions
    """
    pkl_name = "data/schr_dft.pkl"
    df = pd.read_pickle(pkl_name)
    w = df[df["DB"] == "Water2510"]
    r = w.iloc[0]
    g1 = r["Geometry"][r["monAs"]]
    p1 = g1[:, 0]
    c1 = g1[:, 1:]
    g2 = r["Geometry"][r["monBs"]]
    p2 = g2[:, 0]
    c2 = g2[:, 2:]
    qm_tools_aw.tools.write_cartesians_to_xyz(p1, c1, "data/water_distances/water.xyz")
    geom, mAs, mBs, charges = r['Geometry'], r['monAs'], r['monBs'], r['charges']
    print('vals = [')
    geom[mAs, 3] += 0.5
    for i in range(20):
        geom[mAs, 3] -= 0.5
        print('"""')
        qm_tools_aw.tools.print_cartesians_dimer(geom, mAs, mBs, charges)
        print('""",')
    print(']')
    return


if __name__ == "__main__":
    main()
