#!/usr/bin/bash

echo "Starting 2B BJ super DISP TERM fitting"

python3 -u main.py --start_params_d4_key pbe0_2B_BJ_start --powell --fit_dispersion_term --df_path ./plots/ddft_study.pkl --energy_target SAPT_DFT_pbe0_atz_disp
