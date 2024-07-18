#!/usr/bin/bash

echo "Starting 2B BJ ATM CHG super optimization"

# python3 -u main.py --level_theories SAPT_DFT_pbe0_adz_3_IE_pre_d4 --start_params_d4_key pbe0_2B_BJ_ATM_CHG_start --powell --ATM --df_path ./plots/ddft_study.pkl

# python3 -u main.py --level_theories SAPT_DFT_pbe0_adz_3_IE_pre_d4 --start_params_d4_key pbe0_2B_BJ_start --powell --df_path ./plots/ddft_study.pkl

python3 -u main.py --level_theories SAPT_DFT_pbe0_adz_3_IE_pre_d4 --start_params_d4_key pbe0_2B_BJ_start --powell --df_path ./plots/ddft_study.pkl --supramolecular_BJ
