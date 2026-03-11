#!/usr/bin/bash

# python3 -u main.py --level_theories SAPT0_adz_3_IE SAPT0_dz_3_IE  SAPT_DFT_adz_3_IE SAPT_DFT_atz_3_IE SAPT0_jdz_3_IE SAPT0_mtz_3_IE SAPT0_jtz_3_IE SAPT0_atz_3_IE SAPT0_tz_3_IE  --start_params_d4_key HF_OPT_2B_START --intermolecular_BJ > BJ_inter.log 
# python3 -u main.py --df_path plots/basis_study.pkl --level_theories SAPT0_adz_3_IE SAPT_DFT_adz_3_IE  --start_params_d4_key HF_OPT_2B_START --intermolecular_BJ # > BJ_inter.log 
python3 -u main.py --df_path plots/basis_study.pkl --level_theories SAPT0_adz_3_IE --start_params_d4_key sadz --powell # > BJ_inter.log 
# python3 -u main.py --level_theories SAPT0_adz_3_IE  --start_params_d4_key 2B_TT_START4 --intermolecular_TT # > TT_inter.log
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories SAPT_DFT_pbe0_adz_3_IE --start_params_d4_key HF_OPT_2B_START --intermolecular_BJ # > BJ_inter_saptdftd4.log 
#
#
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories SAPT_DFT_pbe0_adz_3_IE --start_params_d4_key 2B_TT_START --intermolecular_TT # > BJ_inter_saptdftd4.log 
# python3 -u main.py --df_path plots/ddft_study.pkl --start_params_d4_key 'SAPT_DFT_pbe0_adz_3_IE_inter_DISP_CORRECTION_START'  --intermolecular_BJ --energy_target 'SAPT_DFT_pbe0_adz_DIFF_SAPT2+3(CCD)DMP2' --fit_dispersion_term
# python3 -u main.py --df_path plots/ddft_study.pkl --start_params_d4_key 'SAPT_DFT_pbe0_adz_3_IE_inter_DISP_CORRECTION_START'  --intermolecular_BJ --energy_target 'SAPT_DFT_pbe0_adz_DIFF_SAPT2+3(CCD)DMP2' --fit_dispersion_term

# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories SAPT_DFT_pbe0_adz_3_IE --start_params_d4_key 2B_TT_START --intermolecular_TT # > BJ_inter_saptdftd4.log 
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_b3lyp_adz_3_IE' --start_params_d4_key 'SAPT_DFT_pbe0_adz_3_IE_inter_START'  --intermolecular_BJ 


# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT0_adz_3_IE' --start_params_d4_key HF_OPT_2B_START --intermolecular_BJ

# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_pbe0_adz_3_IE' 'SAPT_DFT_b3lyp_adz_3_IE' --start_params_d4_key HF_OPT_2B_START --intermolecular_BJ
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_b3lyp_adz_3_IE' --start_params_d3_key 'D3_BJ_START' --intermolecular_BJ_D3
#
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_b3lyp_atz_3_IE' --start_params_d3_key 'D3_BJ_START' --intermolecular_BJ_D3
#
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_pbe0_atz_3_IE' --start_params_d3_key 'D3_BJ_START' --intermolecular_BJ_D3

# Supermolecular
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_b3lyp_adz_3_IE' --start_params_d3_key 'D3_BJ_START' --supermolecular_BJ_D3
#
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_b3lyp_atz_3_IE' --start_params_d3_key 'D3_BJ_START' --supermolecular_BJ_D3
#
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_pbe0_adz_3_IE' --start_params_d3_key 'D3_BJ_START' --supermolecular_BJ_D3
#
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_pbe0_atz_3_IE' --start_params_d3_key 'D3_BJ_START' --supermolecular_BJ_D3

# Can we improve DFT-D4 fitting?
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories "SAPT(DFT) [PBE0] Sum" --start_params_d4_key 'SAPT_DFT_pbe0_adz_3_IE_inter_START' --intermolecular_BJ
# python3 -u main.py --df_path train.pkl --level_theories "SAPT(DFT) [PBE0] Sum" --start_params_d4_key 'SAPT_DFT_pbe0_adz_3_IE_inter_START'  --intermolecular_BJ

# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_b3lyp_adz_3_IE' --start_params_d4_key 'SAPT_DFT_pbe0_adz_3_IE' --powell

# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_b3lyp_adz_3_IE' --start_params_d3_key 'D3_BJ_START' --supermolecular_BJ_D3 --energy_target 'SAPT_DFT_b3lyp_adz_DIFF_SAPT2+3(CCD)DMP2' --fit_dispersion_term
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_b3lyp_atz_3_IE' --start_params_d3_key 'D3_BJ_START' --supermolecular_BJ_D3 --energy_target 'SAPT_DFT_b3lyp_adz_DIFF_SAPT2+3(CCD)DMP2' --fit_dispersion_term
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_pbe0_adz_3_IE' --start_params_d3_key 'D3_BJ_START' --supermolecular_BJ_D3 --energy_target 'SAPT_DFT_b3lyp_adz_DIFF_SAPT2+3(CCD)DMP2' --fit_dispersion_term
# python3 -u main.py --df_path plots/ddft_study.pkl --level_theories 'SAPT_DFT_pbe0_atz_3_IE' --start_params_d3_key 'D3_BJ_START' --supermolecular_BJ_D3 --energy_target 'SAPT_DFT_b3lyp_adz_DIFF_SAPT2+3(CCD)DMP2' --fit_dispersion_term
