import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qm_tools_aw

def water_dimer():
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

def distance_3d(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def read_pyridines() -> None:
    """
    read_pyridines
    """
    p1, c1 = qm_tools_aw.tools.read_xyz_to_pos_carts("data/water_distances/p1.xyz")
    p2, c2 = qm_tools_aw.tools.read_xyz_to_pos_carts("data/water_distances/p2.xyz")
    c1[:, 1] -= 16
    c2[:, 1] = 0
    c = np.concatenate((c1, c2))
    p = np.concatenate((p1, p2))
    qm_tools_aw.tools.write_cartesians_to_xyz(p, c, "data/water_distances/p_together.xyz")
    return c1, c2, p1, p2

def read_waters() -> None:
    """
    read_pyridines
    """
    p1, c1 = qm_tools_aw.tools.read_xyz_to_pos_carts("data/water_distances/w1.xyz")
    p2, c2 = qm_tools_aw.tools.read_xyz_to_pos_carts("data/water_distances/w2.xyz")
    c1[:, 2] += 0.4
    c = np.concatenate((c1, c2))
    p = np.concatenate((p1, p2))
    qm_tools_aw.tools.write_cartesians_to_xyz(p, c, "data/water_distances/w_together.xyz")
    return c1, c2, p1, p2

def main():
    """
    random one time functions
    """
    c1, c2, p1, p2 = read_pyridines()
    print('data = [')
    charges = [[0, 1], [0, 1], [0,1]]
    mAs = range(len(p1))
    mBs = range(len(p1), len(p1) + len(p2))
    for i in range(20):
        c1[:, 1] += 0.5
        print('"""')
        c = np.concatenate((c1, c2))
        p = np.concatenate((p1, p2))
        p3 = p.reshape(-1, 1)
        geom = np.concatenate((p3, c), axis=1)
        qm_tools_aw.tools.print_cartesians_dimer(geom, mAs, mBs, charges)
        print('""",')
    print(']')

    c1, c2, p1, p2 = read_waters()
    mAs = range(len(p1))
    mBs = range(len(p1), len(p1) + len(p2))
    print('data = [')
    for i in range(20):
        c1[:, 1] += 0.5
        print('"""')
        c = np.concatenate((c1, c2))
        p = np.concatenate((p1, p2))
        p3 = p.reshape(-1, 1)
        geom = np.concatenate((p3, c), axis=1)
        qm_tools_aw.tools.print_cartesians_dimer(geom, mAs, mBs, charges)
        print('""",')
    print(']')
    return


if __name__ == "__main__":
    main()
