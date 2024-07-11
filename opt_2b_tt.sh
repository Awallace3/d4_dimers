#!/usr/bin/bash

# python3 -u main.py --level_theories SAPT0_adz_3_IE SAPT0_dz_3_IE  SAPT_DFT_adz_3_IE SAPT_DFT_atz_3_IE SAPT0_jdz_3_IE SAPT0_mtz_3_IE SAPT0_jtz_3_IE SAPT0_atz_3_IE SAPT0_tz_3_IE --start_params_d4_key 2B_TT_START2 --powell_2B_TT_ATM_TT >> out_2b_tt.log 
# python3 -u main.py --level_theories SAPT0_adz_3_IE --start_params_d4_key 2B_TT_START3 --powell_2B_TT_ATM_TT 
# python3 -u main.py --level_theories SAPT0_adz_3_IE --start_params_d4_key 2B_TT_START6 --powell_2B_TT_ATM_TT 
echo "Starting 2B TT super optimization"

# python3 -u main.py --level_theories SAPT0_dz_3_IE SAPT0_jdz_3_IE SAPT0_adz_3_IE SAPT0_tz_3_IE SAPT0_mtz_3_IE SAPT0_jtz_3_IE SAPT0_atz_3_IE SAPT_DFT_adz_3_IE SAPT_DFT_atz_3_IE --start_params_d4_key 2B_TT_START6 --powell_2B_TT_ATM_TT > out_2b_tt_super.log 

echo "Starting 2B TT supra optimization"

python3 -u main.py --level_theories SAPT0_dz_3_IE SAPT0_jdz_3_IE SAPT0_adz_3_IE SAPT0_tz_3_IE SAPT0_mtz_3_IE SAPT0_jtz_3_IE SAPT0_atz_3_IE SAPT_DFT_adz_3_IE SAPT_DFT_atz_3_IE --start_params_d4_key 2B_TT_START6 --supramolecular_TT > out_2b_tt_supra.log 

echo "Starting 2B BJ supra optimization"

python3 -u main.py --level_theories SAPT0_dz_3_IE SAPT0_jdz_3_IE SAPT0_adz_3_IE SAPT0_tz_3_IE SAPT0_mtz_3_IE SAPT0_jtz_3_IE SAPT0_atz_3_IE SAPT_DFT_adz_3_IE SAPT_DFT_atz_3_IE --start_params_d4_key sadz_OPT --supramolecular_BJ > out_2b_bj_supra.log 
