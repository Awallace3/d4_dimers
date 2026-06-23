In this folder, we provide all SAPT and -D3/-D4 interaction energy data for all
computations in this work. Energy units are in Hartree and coordinates are in
Angstrom.

The recommended way to access the data is with python scripts using pandas
dataframes. To access all the data through a single dataframe, load
`./LoS_full_4558.csv` into a python pandas DataFrame:

```py
import pandas as pd
from pprint import pprint as pp

df = pd.read_csv('./LoS_full_4558.csv', index_col=0)

# Then, we can view the contents by first printing the available columns:
pp(df.columns.values.tolist())
```

All units are in atomic units except for coordinates in Angstrom. The
'system_id' column uniquely identifies each molecular dimer and can be used to
combine data from the BFDB (https://vergil.chemistry.gatech.edu/bfdb), which
contains additional levels-of-sapt and MP2 data corresponding to previous work
(https://doi.org/10.1063/5.0275311).

An annotated list of the column labels is below:

```py

['id',         # original DB ID
 'DB',         # database name
 'subset',     # identifies entries with aqz energies
 'system_id',  # unique identifier for each dimer
 'monAs',      # indexing for slicing out monomer A coordinates and atomic_numbers
 'monBs',      # indexing for slicing out monomer B coordinates and atomic_numbers
 'benchmark ref energy',  # estimated CCSD(T)/CBS reference energy
 'coordinates',           # xyz coordinates in Angstrom
 'atomic_numbers',        # atomic numbers of each atom in the dimer
 'dimer_charge',          # charge of the dimer
 'dimer_multiplicity',    # multiplicity of the dimer
 'monA_charge',           # charge of monomer A
 'monA_multiplicity',     # multiplicity of monomer A
 'monB_charge',           # charge of monomer B
 'monB_multiplicity',     # multiplicity of monomer B
 'Benchmark',             # CCSD(T)/CBS energy
 'System Label',          # for dissociation curves, the label identifies system regardless of distance
 'R',                     # for dissociation curves, the equilibrium percent distance between monomers
 'SAPT_DFT_pbe0_atz_total',   # SAPT(PBE0) Component: Total
 'SAPT_DFT_pbe0_atz_elst',    # SAPT(PBE0) Component: Electrostatics
 'SAPT_DFT_pbe0_atz_exch',    # SAPT(PBE0) Component: Exchange
 'SAPT_DFT_pbe0_atz_indu',    # SAPT(PBE0) Component: Induction
 'SAPT_DFT_pbe0_atz_disp',    # SAPT(PBE0) Component: FDDS Dispersion
 'SAPT_DFT_pbe0_atz_3_IE',    # SAPT(PBE0) component sum Elst+Exch+Ind
 'PBE0 IE atz',               # PBE0 interaction energy atz
 'PBE0-D4 IE atz',            # PBE0-D4 interaction energy atz
 'SAPT_DFT_pbe0_atz_dHF',     # SAPT(PBE0) delta_HF
 'SAPT_DFT_pbe0_atz_dDFT',    # SAPT(PBE0) delta_DFT
 'SAPT_DFT_pbe0_atz_D4_IE',   # SAPT(PBE0) supermolecular -D4(ATM) with default Grimme Parameters for PBE0-D4
 'SAPT_DFT_pbe0_atz_DFT_IE',  # SAPT(PBE0) supermolecular DFT IE
 'pbe0_adz_grac_A',           # PBE0/aDZ predicted GRAC shift for monomer A
 'pbe0_adz_grac_B',           # PBE0/aDZ predicted GRAC shift for monomer B
 'b3lyp_adz_grac_A',          # B3LYP/aDZ predicted GRAC shift for monomer A
 'b3lyp_adz_grac_B',          # B3LYP/aDZ predicted GRAC shift for monomer B
 # patterns continue with different method/basis set
 'SAPT_DFT_pbe0_aqz_dHF',
 'SAPT_DFT_pbe0_aqz_dDFT',
 'SAPT_DFT_pbe0_aqz_D4_IE',
 'SAPT_DFT_pbe0_aqz_DFT_IE',
 'SAPT_DFT_b3lyp_adz_dHF',
 'SAPT_DFT_b3lyp_adz_dDFT',
 'SAPT_DFT_b3lyp_adz_D4_IE',
 'SAPT_DFT_b3lyp_adz_DFT_IE',
 'SAPT_DFT_b3lyp_atz_dHF',
 'SAPT_DFT_b3lyp_atz_dDFT',
 'SAPT_DFT_b3lyp_atz_D4_IE',
 'SAPT_DFT_b3lyp_atz_DFT_IE',
 'SAPT_DFT_b3lyp_aqz_dHF',
 'SAPT_DFT_b3lyp_aqz_dDFT',
 'SAPT_DFT_b3lyp_aqz_D4_IE',
 'SAPT_DFT_b3lyp_aqz_DFT_IE',
 'charges', # charge/multiplicity info for dimer
 'SAPT_DFT_pbe0_adz_dHF',
 'SAPT_DFT_pbe0_adz_dDFT',
 'SAPT_DFT_pbe0_adz_D4_IE',
 'SAPT_DFT_pbe0_adz_DFT_IE',
 'SAPT_DFT_pbe0_atz_dHF',
 'SAPT_DFT_pbe0_atz_dDFT',
 'SAPT_DFT_pbe0_atz_D4_IE',
 'SAPT_DFT_pbe0_atz_DFT_IE',
 'pbe0_atz_grac_A',
 'pbe0_atz_grac_B',
 'SAPT_DFT_pbe0_adz_total',
 'SAPT_DFT_pbe0_adz_d4_total',
 'E_R_eq_elst_aqz',
 'E_R_eq_elst_atz',
 'E_R_eq_exch_aqz',
 'E_R_eq_exch_atz',
 'E_R_eq_ind_aqz',
 'E_R_eq_ind_atz',
 'E_R_eq',
 'MP2_atqz',
 'identifiers',
 '-D ENERGY adz',
 '-D ENERGY atz',
 'NUCLEAR REPULSION ENERGY adz',
 'NUCLEAR REPULSION ENERGY aqz',
 'NUCLEAR REPULSION ENERGY atz',
 'NUCLEAR REPULSION ENERGY jdz',
 'SAPT0 DISP ENERGY adz',
 'SAPT0 DISP ENERGY aqz',
 'SAPT0 DISP ENERGY atz',
 'SAPT0 DISP ENERGY jdz',
 'SAPT0 ELST ENERGY adz',
 'SAPT0 ELST ENERGY aqz',
 'SAPT0 ELST ENERGY atz',
 'SAPT0 ELST ENERGY jdz',
 'SAPT0 EXCH ENERGY adz',
 'SAPT0 EXCH ENERGY aqz',
 'SAPT0 EXCH ENERGY atz',
 'SAPT0 EXCH ENERGY jdz',
 'SAPT0 IND ENERGY adz',
 'SAPT0 IND ENERGY aqz',
 'SAPT0 IND ENERGY atz',
 'SAPT0 IND ENERGY jdz',
 'SAPT0 TOTAL ENERGY adz',
 'SAPT0 TOTAL ENERGY aqz',
 'SAPT0 TOTAL ENERGY atz',
 'SAPT0 TOTAL ENERGY jdz',
 'SAPT2+3(CCD) DISP ENERGY adz',
 'SAPT2+3(CCD) DISP ENERGY aqz',
 'SAPT2+3(CCD) DISP ENERGY atz',
 'SAPT2+3(CCD) DISP ENERGY jdz',
 'SAPT2+3(CCD) ELST ENERGY adz',
 'SAPT2+3(CCD) ELST ENERGY aqz',
 'SAPT2+3(CCD) ELST ENERGY atz',
 'SAPT2+3(CCD) ELST ENERGY jdz',
 'SAPT2+3(CCD) EXCH ENERGY adz',
 'SAPT2+3(CCD) EXCH ENERGY aqz',
 'SAPT2+3(CCD) EXCH ENERGY atz',
 'SAPT2+3(CCD) EXCH ENERGY jdz',
 'SAPT2+3(CCD) IND ENERGY adz',
 'SAPT2+3(CCD) IND ENERGY aqz',
 'SAPT2+3(CCD) IND ENERGY atz',
 'SAPT2+3(CCD) IND ENERGY jdz',
 'SAPT2+3(CCD) TOTAL ENERGY adz',
 'SAPT2+3(CCD) TOTAL ENERGY aqz',
 'SAPT2+3(CCD) TOTAL ENERGY atz',
 'SAPT2+3(CCD) TOTAL ENERGY jdz',
 'D3-ML',
 'SAPT_DFT_D4_pbe0_adz_total',
 'SAPT_DFT_pbe0_adz_elst',
 'SAPT_DFT_pbe0_adz_exch',
 'SAPT_DFT_pbe0_adz_indu',
 'SAPT_DFT_pbe0_adz_disp',
 'SAPT_DFT_pbe0_adz_3_IE',
 'SAPT_DFT_pbe0_adz_d4_disp',
 'PBE0 IE adz',
 'PBE0-D4 IE adz',
 'SAPT_DFT_D4_pbe0_aqz_total',
 'SAPT_DFT_pbe0_aqz_total',
 'SAPT_DFT_pbe0_aqz_elst',
 'SAPT_DFT_pbe0_aqz_exch',
 'SAPT_DFT_pbe0_aqz_indu',
 'SAPT_DFT_pbe0_aqz_disp',
 'SAPT_DFT_pbe0_aqz_3_IE',
 'SAPT_DFT_pbe0_aqz_d4_disp',
 'PBE0 IE aqz',
 'PBE0-D4 IE aqz',
 'SAPT_DFT_D4_b3lyp_adz_total',
 'SAPT_DFT_b3lyp_adz_total',
 'SAPT_DFT_b3lyp_adz_elst',
 'SAPT_DFT_b3lyp_adz_exch',
 'SAPT_DFT_b3lyp_adz_indu',
 'SAPT_DFT_b3lyp_adz_disp',
 'SAPT_DFT_b3lyp_adz_3_IE',
 'SAPT_DFT_b3lyp_adz_d4_disp',
 'B3LYP IE adz',
 'B3LYP-D4 IE adz',
 'SAPT_DFT_D4_b3lyp_atz_total',
 'SAPT_DFT_b3lyp_atz_total',
 'SAPT_DFT_b3lyp_atz_elst',
 'SAPT_DFT_b3lyp_atz_exch',
 'SAPT_DFT_b3lyp_atz_indu',
 'SAPT_DFT_b3lyp_atz_disp',
 'SAPT_DFT_b3lyp_atz_3_IE',
 'SAPT_DFT_b3lyp_atz_d4_disp',
 'B3LYP IE atz',
 'B3LYP-D4 IE atz',
 'SAPT_DFT_D4_b3lyp_aqz_total',
 'SAPT_DFT_b3lyp_aqz_total',
 'SAPT_DFT_b3lyp_aqz_elst',
 'SAPT_DFT_b3lyp_aqz_exch',
 'SAPT_DFT_b3lyp_aqz_indu',
 'SAPT_DFT_b3lyp_aqz_disp',
 'SAPT_DFT_b3lyp_aqz_3_IE',
 'SAPT_DFT_b3lyp_aqz_d4_disp',
 'B3LYP IE aqz',
 'B3LYP-D4 IE aqz',
 'SAPT0_adz_total',
 'SAPT0_adz_elst',
 'SAPT0_adz_exch',
 'SAPT0_adz_indu',
 'SAPT0_adz_disp',
 'SAPT0_adz_3_IE',
 'SAPT0_atz_3_IE',
 'SAPT0_atz_total',
 'SAPT0_atz_elst',
 'SAPT0_atz_exch',
 'SAPT0_atz_indu',
 'SAPT0_atz_disp',
 'SAPT0_aqz_3_IE',
 'SAPT0_aqz_total',
 'SAPT0_aqz_elst',
 'SAPT0_aqz_exch',
 'SAPT0_aqz_indu',
 'SAPT0_aqz_disp',
 '-D4 (SAPT0_adz_3_IE)',  # SAPT0-D4(S) dispersion energies
 '-D4 (SAPT0_atz_3_IE)',  # SAPT0-D4(S) dispersion energies
 '-D4 (SAPT0_aqz_3_IE)',  # SAPT0-D4(S) dispersion energies
 'SAPT0_adz_d4',
 'SAPT0_atz_d4',
 'SAPT0_aqz_d4',
 'SAPT0-D4/aDZ',
 'SAPT(DFT)D3-ML TOTAL ENERGY adz',
 'SAPT(DFT)D3-ML TOTAL ENERGY atz',
 'SAPT(DFT)D3-ML TOTAL ENERGY aqz',
 'D3-ML DISP ENERGY adz',
 'D3-ML DISP ENERGY atz',
 'D3-ML DISP ENERGY aqz',
 '-D4 (SAPT_DFT_pbe0_adz_3_IE)',  # SAPT(DFT)-D4(S) dispersion energies
 '-D4 (SAPT_DFT_pbe0_atz_3_IE)',  # SAPT(DFT)-D4(S) dispersion energies
 '-D4 (SAPT_DFT_pbe0_aqz_3_IE)',  # SAPT(DFT)-D4(S) dispersion energies
 '-D4 (HF)',
 '-D4 (HF_ATM)',
 '-D4 (SAPT_DFT_pbe0_adz_3_IE_inter)',
 'SAPT(DFT) [PBE0] ELST ENERGY adz',
 'SAPT(DFT) [PBE0] EXCH ENERGY adz',
 'SAPT(DFT) [PBE0] IND ENERGY adz',
 'SAPT(DFT) [PBE0] DISP ENERGY adz',
 'PBE0-D4 DISP ENERGY adz',
 'SAPT(DFT) [PBE0] Sum',
 '-D4 (SAPT_DFT_b3lyp_adz_3_IE_inter)',
 'SAPT(PBE0)D3-ML TOTAL ENERGY adz',
 'SAPT(PBE0)D3-ML TOTAL ENERGY atz',
 'SAPT(PBE0)D3-ML TOTAL ENERGY aqz',
 'SAPT(B3LYP)D3-ML TOTAL ENERGY adz',
 'SAPT(B3LYP)D3-ML TOTAL ENERGY atz',
 'SAPT(B3LYP)D3-ML TOTAL ENERGY aqz',
 '-D4 (SAPT0_adz_3_IE_2B_BJ_inter)',
 'SAPT0-D4 TOTAL ENERGY adz',
 'SAPT0-D4 TOTAL ENERGY atz',
 'SAPT0-D4 TOTAL ENERGY aqz',
 'SAPT0-D4 INTER TOTAL ENERGY adz',
 'SAPT0-D4 INTER TOTAL ENERGY atz',
 'SAPT0-D4 INTER TOTAL ENERGY aqz',
 'SAPT0-D4 INTER DISP ENERGY adz',
 'SAPT0-D4 INTER DISP ENERGY atz',
 'SAPT0-D4 INTER DISP ENERGY aqz',
 'SAPT(PBE0)-D4 INTER DISP ENERGY',
 'SAPT(PBE0)-D4 INTER DISP ENERGY adz',
 'SAPT(PBE0)-D4 INTER DISP ENERGY atz',
 'SAPT(PBE0)-D4 INTER DISP ENERGY aqz',
 'SAPT(PBE0)-D4 INTER TOTAL ENERGY adz',
 'SAPT(PBE0)-D4 INTER TOTAL ENERGY atz',
 'SAPT(PBE0)-D4 INTER TOTAL ENERGY aqz',
 'SAPT(B3LYP)-D4 INTER DISP ENERGY',
 'SAPT(B3LYP)-D4 INTER DISP ENERGY adz',
 'SAPT(B3LYP)-D4 INTER DISP ENERGY atz',
 'SAPT(B3LYP)-D4 INTER DISP ENERGY aqz',
 'SAPT(B3LYP)-D4 INTER TOTAL ENERGY adz',
 'SAPT(B3LYP)-D4 INTER TOTAL ENERGY atz',
 'SAPT(B3LYP)-D4 INTER TOTAL ENERGY aqz',
 'SAPT(PBE0)-D3 INTER DISP ENERGY',
 'SAPT(PBE0)-D3 INTER DISP ENERGY adz',
 'SAPT(PBE0)-D3 INTER DISP ENERGY atz',
 'SAPT(PBE0)-D3 INTER DISP ENERGY aqz',
 'SAPT(PBE0)-D3 INTER TOTAL ENERGY adz',
 'SAPT(PBE0)-D3 INTER TOTAL ENERGY atz',
 'SAPT(PBE0)-D3 INTER TOTAL ENERGY aqz',
 'SAPT_DFT_pbe0_adz_D3_IE',
 'SAPT_DFT_pbe0_atz_D3_IE',
 'SAPT_DFT_pbe0_aqz_D3_IE',
 'PBE0-D3 DISP ENERGY',
 'PBE0-D3 DISP ENERGY adz',
 'PBE0-D3 DISP ENERGY atz',
 'PBE0-D3 DISP ENERGY aqz',
 'PBE0-D3 IE adz',
 'PBE0-D3 IE atz',
 'PBE0-D3 IE aqz',
 'SAPT_DFT_b3lyp_adz_D3_IE',
 'SAPT_DFT_b3lyp_atz_D3_IE',
 'SAPT_DFT_b3lyp_aqz_D3_IE',
 'B3LYP-D3 DISP ENERGY adz',
 'B3LYP-D3 DISP ENERGY atz',
 'B3LYP-D3 DISP ENERGY aqz',
 'B3LYP-D3 IE adz',
 'B3LYP-D3 IE atz',
 'B3LYP-D3 IE aqz',
 'SAPT(PBE0)-D3 SUPER DISP ENERGY',
 'SAPT(PBE0)-D3 SUPER DISP ENERGY adz',
 'SAPT(PBE0)-D3 SUPER DISP ENERGY atz',
 'SAPT(PBE0)-D3 SUPER DISP ENERGY aqz',
 'SAPT(PBE0)-D3 SUPER TOTAL ENERGY adz',
 'SAPT(PBE0)-D3 SUPER TOTAL ENERGY atz',
 'SAPT(PBE0)-D3 SUPER TOTAL ENERGY aqz',
 'SAPT(B3LYP)-D3 INTER DISP ENERGY',
 'SAPT(B3LYP)-D3 INTER DISP ENERGY adz',
 'SAPT(B3LYP)-D3 INTER DISP ENERGY atz',
 'SAPT(B3LYP)-D3 INTER DISP ENERGY aqz',
 'SAPT(B3LYP)-D3 INTER TOTAL ENERGY adz',
 'SAPT(B3LYP)-D3 INTER TOTAL ENERGY atz',
 'SAPT(B3LYP)-D3 INTER TOTAL ENERGY aqz',
 'SAPT(B3LYP)-D3 SUPER DISP ENERGY',
 'SAPT(B3LYP)-D3 SUPER DISP ENERGY adz',
 'SAPT(B3LYP)-D3 SUPER DISP ENERGY atz',
 'SAPT(B3LYP)-D3 SUPER DISP ENERGY aqz',
 'SAPT(B3LYP)-D3 SUPER TOTAL ENERGY adz',
 'SAPT(B3LYP)-D3 SUPER TOTAL ENERGY atz',
 'SAPT(B3LYP)-D3 SUPER TOTAL ENERGY aqz',
 'PBE0-D3 TOTAL ENERGY adz',
 'B3LYP-D3 TOTAL ENERGY adz',
 'PBE0-D3 TOTAL ENERGY atz',
 'B3LYP-D3 TOTAL ENERGY atz',
 'PBE0-D3 TOTAL ENERGY aqz',
 'B3LYP-D3 TOTAL ENERGY aqz']
```
