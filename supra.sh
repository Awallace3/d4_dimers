#!/usr/bin/bash

python3 -u main.py --level_theories SAPT0_adz_3_IE SAPT0_dz_3_IE  SAPT_DFT_adz_3_IE SAPT_DFT_atz_3_IE SAPT0_jdz_3_IE SAPT0_mtz_3_IE SAPT0_jtz_3_IE SAPT0_atz_3_IE SAPT0_tz_3_IE  --start_params_d4_key HF_OPT_2B_START --supramolecular_BJ > BJ_supra.log 
# python3 -u main.py --level_theories SAPT0_adz_3_IE  --start_params_d4_key 2B_TT_START4 --supramolecular_TT # > TT_supra.log
