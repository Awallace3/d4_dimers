#! /usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import scipy.optimize as opt
import time
from src.tools import print_cartesians


#parse args
parser = argparse.ArgumentParser(description="SAPT-D fitter")
parser.add_argument('-d', '--damp', type=str, help="Damping function")
parser.add_argument('-m', '--metric', type=str, help="fitting metric")
args = parser.parse_args()

class ExitEarly(Exception):
    """Terminating after one cycle"""


def get_db(reffile):
    # Get the master database
    master = pd.read_pickle(reffile)

    master = master[master['DB'] != 'PCONF']
    master = master[master['DB'] != 'SCONF']
    master = master[master['DB'] != 'ACONF']
    master = master[master['DB'] != 'CYCONF']
    weights = []
    dref = []
    bm = []
    df = pd.read_csv("SAPT-D3-Refit-Data.csv")
    for idx, item in master.iterrows():
        db = item['DB']
        sys = item['System']
        r = item['R']

        try:
            #ret = df.loc[(df['DB'] == db) & (df['System'] == sys) & (df['R'] == r), ['Benchmark', 'Weight', 'HF-CP-qzvp-BJ-F']].to_numpy()[0]
            ret = df.loc[(df['System'] == sys) & (abs(df['Benchmark'] - item['Benchmark']) < 1e-6), ['Benchmark', 'Weight', 'HF-CP-qzvp-BJ-F', 'HF-CP-qzvp-Zero-F']].to_numpy()[0]
        except:
            print(db, sys, r, item['Benchmark'])
            print("Exiting")
            exit()

        bm.append(ret[0])
        weights.append(ret[1])
        dref.append(ret[2])


    master['weights'] = weights
    master['dref'] = dref
    master['bm'] = bm

    #training = master[master['Fitset']==True]
    training = master
    ntrain = len(training)
    return training

def compute_bj(params, d3data):
    s8, a1, a2 = params
    energy = 0.0
    for pair in d3data:
        atom1, atom2, R, R0, C6, C8 = pair

        R0 = np.sqrt(C8/C6)
        energy += C6 / (R**6.0 + (a1*R0 + a2)**6.0)
        energy += s8*C8 / (R**8.0 + (a1*R0 + a2)**8.0)

    energy *= -1.0 * 627.5095
    return energy

def compute_zero(params, d3data):
    # Original CHG damping
    # Fitting paramters are s8 and sr6

    s8, sr6 = params
    energy = 0.0
    for pair in d3data:
        atom1, atom2, R, R0, C6, C8 = pair

        energy += C6 / (R**6 * (1 + 6 * (R/(sr6*R0))**-14.0 ))
        energy += (s8*C8) / (R**8 * (1 + 6 * (R/(R0))**-16.0 ))

    energy *= -1.0 * 627.5095
    return energy

def compute_zero3_b(params, d3data):
    # CHG implementation from Daniel
    # Strange reappearance of sr6 in the denom
    # Fitting paramters are s8 and sr6

    s8, sr6, beta = params
    energy = 0.0
    for pair in d3data:
        atom1, atom2, R, R0, C6, C8 = pair

        energy += C6 / (R**6 * (1 + 6 * (R/(sr6*R0) + beta*R0*sr6)**-14.0 ))
        energy += (s8*C8) / (R**8 * (1 + 6 * (R/(R0) + beta*R0)**-16.0 ))

    energy *= -1.0 * 627.5095
    return energy

def compute_zero3_b2(params, d3data):
    # Original CHG damping
    # In my view, the 'correct' CHG implementation
    # Fitting paramters are s8 and sr6

    s8, sr6, beta = params
    energy = 0.0
    for pair in d3data:
        atom1, atom2, R, R0, C6, C8 = pair

        energy += C6 / (R**6 * (1 + 6 * (R/(sr6*R0) + beta*R0)**-14.0 ))
        energy += (s8*C8) / (R**8 * (1 + 6 * (R/(R0) + beta*R0)**-16.0 ))

    energy *= -1.0 * 627.5095
    return energy

def compute_zero3_s8(params, d3data):
    # Original CHG damping
    # Why introduce beta, when there's already sr8 we could use?
    # Fitting paramters are s8 and sr6

    s8, sr6, sr8 = params
    energy = 0.0
    for pair in d3data:
        atom1, atom2, R, R0, C6, C8 = pair

        energy += C6 / (R**6 * (1 + 6 * (R/(sr6*R0))**-14.0 ))
        energy += (s8*C8) / (R**8 * (1 + 6 * (R/(sr8*R0))**-16.0 ))

    energy *= -1.0 * 627.5095
    return energy



def compute_int_energy(params):

    damp = args.damp.upper()
    # Loop over each dimer in the training set
    # training = get_db('master-0.85.pkl')
    training = get_db('master-regen.pkl')
    ntrain = len(training)
    train_weights = np.zeros(ntrain)
    res = np.zeros(ntrain)
    d3s = np.zeros(ntrain)
    n = 0
    for idx,item in training.iterrows():
        d3data = item['D3Data']

        if damp == 'BJ':
            en = compute_bj(abs(params), d3data)
        if damp == 'ZERO':
            en = compute_zero(abs(params), d3data)
        if damp == 'ZERO3-B':
            en = compute_zero3_b(abs(params), d3data)
        if damp == 'ZERO3-B2':
            en = compute_zero3_b2(abs(params), d3data)
        if damp == 'ZERO3-S8':
            en = compute_zero3_s8(abs(params), d3data)
        d3s[n] = en
        dhf = item['HF INTERACTION ENERGY']
        sdd = en + dhf
        ref = item['Benchmark']
        res[n] = ref - sdd
        train_weights[n] = item['weights']
        n += 1

    training['d3'] = d3s
    training['diff'] = res

    df = training.copy()
    df["diff_abs"] = df['diff'].abs()
    df = df.sort_values(by=["diff_abs"], ascending=False)
    # print(df[["Benchmark", "HF INTERACTION ENERGY", "d3", "diff", "diff_abs"]].head(30))
    # print(df.iloc[0][["DB", "System", "Benchmark", "HF INTERACTION ENERGY", "diff", "diff_abs"]])


    rmse = np.sqrt(np.mean(np.square(res)))
    wrmse = np.sqrt(np.mean(np.divide(np.square(res),np.square(train_weights) )))
    wmse = np.mean(np.divide(np.square(res),np.square(train_weights) ))
    wmure = np.mean(np.divide(abs(res),train_weights)) * 100

    mae = df["diff"].abs().sum() / len(df["diff"])
    rmse = (df["diff"] ** 2).mean() ** 0.5
    max_e = df["diff"].abs().max()

    # print(params, rmse, wrmse, wmure)
    df = df.sort_values(by=["diff_abs"], ascending=False)
    df = df.reset_index(drop=False)
    hf_key = "HF INTERACTION ENERGY"
    print("        1. MAE  = %.4f" % mae)
    print("        2. RMSE = %.4f" % rmse)
    print("        3. MAX  = %.4f" % max_e)
    print(
        df[
            [
                "index",
                "DB",
                "Benchmark",
                hf_key,
                "diff",
                "diff_abs",
            ]
        ].head(30)
    )
    for i in range(10):
        print(f"\nMol {i}")
        print(df.iloc[i][[
                "index",
                "DB",
                "Benchmark",
                hf_key,
                "diff",
                "diff_abs",
            ]])
        print("\nCartesians")
        print_cartesians(df.iloc[i]["Geometry"])
    training.to_pickle("jeff.pkl")

    metric = args.metric
    raise ExitEarly()
    if args.metric.upper() == 'RMSE':
        return rmse
    if args.metric.upper() == 'WRMSE':
        return wrmse

#BJ
#init = [0.8,0.2,5.0]
damp = args.damp.upper()
if damp == "BJ":
    #init = [1.49243587, 0.25149731, 3.84835139]
    # init = [0.9171, 0.3385, 2.883]
    init = [0.713190, 0.079541, 3.627854]

#Zero
if damp == "ZERO":
    init = [1.8,1.0]

#zero3-2
if damp == "ZERO3-B":
    init = [1.253384, 1.680858, 0.05286]

if damp == "ZERO3-B2":
    init = [1.48089955, 1.47393631, 0.05417519]

# zero3-s8
if damp == "ZERO3-S8":
    init = [0.97076554, 0.72900611, 1.0]

print(f"Damping: {damp}")
print(f"Metric: {args.metric.upper()}")
ret = opt.minimize(compute_int_energy, init, method='powell')

print(ret)


