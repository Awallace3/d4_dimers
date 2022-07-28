#! /usr/bin/env python

import numpy as np
import pandas as pd
import scipy.optimize as opt
import argparse

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

    training = master
    #training = master[master['Fitset']==True]
    ntrain = len(training)
    
    return training

# Make folds
# returns nfold lists of 
def get_folds(nfold, ntrain):
    folds = []

    for f in range(nfold):
        f_def = []
        for n in range(ntrain):
            if (n % nfold) == f:
                f_def.append(False)
            else:
                f_def.append(True)

        folds.append(f_def)
    return folds

def compute_bj(params, d3data):
    s8, a1, a2 = params
    energy = 0.0
    for pair in d3data:
        atom1, atom2, R, R0, C6, C8 = pair

        R0 = np.sqrt(C8/C6)
        energy += C6 / (R**6.0 + (a1*R0 + a2)**6.0)  
        energy += s8*C8 / (R**8.0 + (a1*R0 + a2)**8.0)  

    energy *= -1.0 * 627.509
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

    energy *= -1.0 * 627.509
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

    energy *= -1.0 * 627.509
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

    energy *= -1.0 * 627.509
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

    energy *= -1.0 * 627.509
    return energy

def compute_int_energy(params, training, print_stat=False):

    damp = args.damp.upper()
    # Loop over each dimer in the training set
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
        sdd = en + dhf 
    
        ref = item['Benchmark']

        res[n] = ref - sdd
        train_weights[n] = item['weights']
        n += 1
 
    mae = np.average(np.abs(res))
    rmse = np.sqrt(np.mean(np.square(res)))
    maxe = np.max(np.abs(res))
    wrmse = np.sqrt(np.mean(np.divide(np.square(res),np.square(train_weights) )))
    wmure = np.mean(np.divide(abs(res),train_weights)) * 100

    if print_stat:
        print(params, mae, rmse, maxe, wrmse, wmure)
    
    metric = args.metric
    if args.metric.upper() == 'RMSE':
        return rmse
    if args.metric.upper() == 'WRMSE':
        return wrmse


master = pd.read_pickle('../master-regen.pkl')
ntrain = len(master)

print(f"Damping function: {args.damp}")
print(f"Error metric: {args.metric}")
        
# get folds, we'll do 5-fold cv
folds = get_folds(5, ntrain)
master = get_db('../master-regen.pkl')
errors = []
for n,fold in enumerate(folds):
    # need to update Fitset column using fold definitions
    print(f"Fold {n}")
    master['Fitset'] = fold 
    training = master[master['Fitset']==True] 
    testing = master[master['Fitset']==False] 
    print(f"Training: {len(training)}")
    print(f"Testing: {len(testing)}")

    #BJ
    damp = args.damp.upper()
    if damp == 'BJ':
        init = [0.9171, 0.3385, 2.883]
    if damp == "ZERO":
        init = [1.8,1.0]
    if damp == "ZERO3-B":
        init = [1.253384, 1.680858, 0.05286]
    if damp == "ZERO3-B2":
        init = [0.885517,1.383214,0.07548]
    if damp == "ZERO3-S8":
        init = [0.97076554, 0.72900611, 1.0]


    ret = opt.minimize(compute_int_energy, init, training, method='powell')
    print(ret.x, ret.fun, ret.success) 

    params = ret.x
    f_rmse = compute_int_energy(params, testing, print_stat=True) 
    errors.append(f_rmse)

avr = 0.0
for er in errors:
    avr += er
print(f"Average RMSE across 5 folds: {avr/5.0}")

 
    

