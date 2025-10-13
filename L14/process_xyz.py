
import pandas as pd
import glob
import os

# Get all xyz files
xyz_files = glob.glob("**/*.xyz", recursive=True)

data = []
for xyz_file in xyz_files:
    # Get molecule name from file name
    molecule_name = os.path.splitext(os.path.basename(xyz_file))[0]
    
    with open(xyz_file, 'r') as f:
        xyz_content = f.read()
    
    data.append([molecule_name, xyz_content])

# Create pandas DataFrame
df = pd.DataFrame(data, columns=["Molecule", "XYZ"])

# Save DataFrame to csv
df.to_csv("l14_molecules.csv", index=False)

print("Successfully created l14_molecules.csv")
