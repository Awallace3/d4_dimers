#!/usr/bin/bash

# python3 -u main.py --level_theories SAPT0_adz_3_IE SAPT0_dz_3_IE  SAPT_DFT_adz_3_IE SAPT_DFT_atz_3_IE SAPT0_jdz_3_IE SAPT0_mtz_3_IE SAPT0_jtz_3_IE SAPT0_atz_3_IE SAPT0_tz_3_IE  --start_params_d4_key HF_OPT_2B_START --supramolecular_BJ > BJ_supra.log 
# python3 -u main.py --level_theories SAPT0_adz_3_IE  --start_params_d4_key 2B_TT_START4 --supramolecular_TT # > TT_supra.log
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories SAPT_DFT_pbe0_adz_3_IE --start_params_d4_key HF_OPT_2B_START --supramolecular_BJ # > BJ_supra_saptdftd4.log 
#
#
python3 -u main.py --df_path plots/ddft_study.pkl --level_theories SAPT_DFT_pbe0_adz_3_IE --start_params_d4_key 2B_TT_START --supramolecular_TT # > BJ_supra_saptdftd4.log 
# python3 -u main.py --df_path plots/ddft_study.pkl --start_params_d4_key 'SAPT_DFT_pbe0_adz_3_IE_supra_DISP_CORRECTION_START'  --supramolecular_BJ --energy_target 'SAPT_DFT_pbe0_adz_DIFF_SAPT2+3(CCD)DMP2' --fit_dispersion_term
