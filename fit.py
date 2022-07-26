#! /usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import scipy.optimize as opt
import time


#parse args
parser = argparse.ArgumentParser(description="SAPT-D fitter")
parser.add_argument('-d', '--damp', type=str, help="Damping function")
parser.add_argument('-m', '--metric', type=str, help="fitting metric")
args = parser.parse_args()



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
    print(ntrain)
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
    #training = get_db('master-0.85.pkl')
    training = get_db('master-regen.pkl')
    ntrain = len(training)
    train_weights = np.zeros(ntrain)
    res = np.zeros(ntrain)
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

        dhf = item['HF INTERACTION ENERGY']
        print("dhf", dhf)
        print("en", en)
        sdd = en + dhf
        print("sdd", sdd)
        ref = item['Benchmark']
        print("ref", ref)
        print("res", ref - sdd)
        return

        res[n] = ref - sdd
        train_weights[n] = item['weights']
        n += 1

    rmse = np.sqrt(np.mean(np.square(res)))
    wrmse = np.sqrt(np.mean(np.divide(np.square(res),np.square(train_weights) )))
    wmse = np.mean(np.divide(np.square(res),np.square(train_weights) ))
    wmure = np.mean(np.divide(abs(res),train_weights)) * 100


    print(params, rmse, wrmse, wmure)

    metric = args.metric
    if args.metric.upper() == 'RMSE':
        return rmse
    if args.metric.upper() == 'WRMSE':
        return wrmse

#BJ
#init = [0.8,0.2,5.0]
damp = args.damp.upper()
if damp == "BJ":
    #init = [1.49243587, 0.25149731, 3.84835139]
    init = [0.9171, 0.3385, 2.883]

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


