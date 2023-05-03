#!/bin/bash

for i in "HF_dz" "HF_jdz" "HF_adz" "HF_tz" "HF_atz"
do
    sed "s/TAG/$i/" main.py > opt.py
    python3 -u opt.py > outs/$i.out & disown
    sleep 1
done
