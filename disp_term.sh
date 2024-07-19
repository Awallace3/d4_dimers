#!/usr/bin/bash

echo "Starting 2B BJ super DISP TERM fitting"

# python3 -u main.py --start_params_d4_key pbe0_2B_BJ_start --powell --fit_dispersion_term --df_path ./plots/ddft_study.pkl --energy_target SAPT_DFT_pbe0_atz_disp
#
python3 -u main.py --start_params_d4_key 'SAPT_DFT_pbe0_adz_disp_targeting_SAPT2+3(CCD)dMP2_start' --powell --fit_dispersion_term --df_path ./plots/ddft_study.pkl --energy_target 'SAPT_DFT_pbe0_adz_DIFF_SAPT2+3(CCD)DMP2' > disp_term_adz.log
# python3 -u main.py --start_params_d4_key 'SAPT_DFT_pbe0_atz_DIFF_SAPT2+3(CCD)DMP2' --powell --fit_dispersion_term --df_path ./plots/ddft_study.pkl --energy_target SAPT_DFT_pbe0_atz_disp
