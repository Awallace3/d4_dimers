
import pandas as pd
import re

df = pd.read_csv("l14_molecules.csv")

energies = []
for xyz in df["XYZ"]:
    match = re.search(r"(canonical CCSD\(T\)/CBS|DLPNO-CCSD\(T1\)/CBS \(VeryTightPNO\)) = ([-+]?\d*\.\d+|\d+)", xyz)
    if match:
        energies.append(float(match.group(2)))
    else:
        energies.append(None)

df["CCSD(T)/CBS"] = energies

df.to_csv("l14_molecules_with_energies.csv", index=False)

print("Successfully created l14_molecules_with_energies.csv")
