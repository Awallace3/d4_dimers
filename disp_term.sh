#!/usr/bin/bash

echo "Starting 2B BJ super DISP TERM fitting"

# python3 -u main.py --start_params_d4_key pbe0_2B_BJ_start --powell --fit_dispersion_term --df_path ./plots/ddft_study.pkl --energy_target SAPT_DFT_pbe0_atz_disp
#
# echo "SAPT_DFT_adz" > disp_term.log
#
# python3 -u main.py --start_params_d4_key 'SAPT_DFT_pbe0_adz_disp_targeting_SAPT2+3(CCD)dMP2_start' --powell --fit_dispersion_term --df_path ./plots/ddft_study.pkl --energy_target 'SAPT_DFT_pbe0_adz_DIFF_SAPT2+3(CCD)DMP2' >> disp_term.log
#
# echo "SAPT_DFT_atz" >> disp_term.log
#
# python3 -u main.py --start_params_d4_key 'SAPT_DFT_pbe0_adz_disp_targeting_SAPT2+3(CCD)dMP2_start' --powell --fit_dispersion_term --df_path ./plots/ddft_study.pkl --energy_target 'SAPT_DFT_pbe0_atz_DIFF_SAPT2+3(CCD)DMP2' >> disp_term.log
#
# echo "Fitting -D4 to SAPT2+3(CCD)DMP2 DISP ENERGY" >> disp_term.log
#
# python3 -u main.py --start_params_d4_key 'HF_OPT_2B_START' --powell --fit_dispersion_term --df_path ./plots/ddft_study.pkl --energy_target 'SAPT2+3(CCD)DMP2 DISP ENERGY' >> disp_term.log

# echo "Fitting -D4 BJ+CHG to SAPT2+3(CCD)DMP2 DISP ENERGY" >> disp_term.log
#
# python3 -u main.py --start_params_d4_key 'HF_OPT_2B_START' --powell --ATM --fit_dispersion_term --df_path ./plots/ddft_study.pkl --energy_target 'SAPT2+3(CCD)DMP2 DISP ENERGY' >> disp_term.log

echo "Fitting SAPT(DFT)-D4 ENERGY" >> saptdftd4.log

# python3 -u main.py --level_theories SAPT_DFT_pbe0_adz_3_IE SAPT_DFT_pbe0_atz_3_IE --start_params_d4_key 'SAPT_DFT_pbe0_IE_start' --powell --df_path ./plots/ddft_study.pkl >> saptdftd4.log
python3 -u main.py --level_theories SAPT_DFT_pbe0_aqz_3_IE --start_params_d4_key 'SAPT_DFT_pbe0_IE_start' --powell --df_path ./plots/ddft_study.pkl >> saptdftd4.log
