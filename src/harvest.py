import pandas as pd
import subprocess


def ssi_bfdb_data(df: pd.DataFrame, basis: str = "adz"):
    """
    collects data for SSI database
    """
    url = "https://raw.githubusercontent.com/loriab/qcdb/master/data/SSI_pt2misc.py"
    subprocess.call("curl '%s' > ssi_aug_cc_pvdz_ie.py" % url, shell=True)
    df2 = df.loc[df["DB"] == "SSI"]
    ind1 = df.index[df["DB"] == "SSI"]
    for i in ind1:
        s = df.loc[i, "System"]
        s = s.replace("Residue ", "")
        s = s.replace(" and ", "-")
        s = s.replace(" interaction No. ", "-")
        cmd = (
            """grep '%s' ssi_aug_cc_pvdz_ie.py | grep "'HF'" | grep "'CP'" | grep "'%s'" | sed 's/=/ /g' | sed 's/)//g' | awk '{print $(NF)}'"""
            % (s, basis)
        )
        v = subprocess.run(cmd, shell=True, capture_output=True)
        v = float(v.stdout)
    return
