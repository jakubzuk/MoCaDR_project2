import json 

import numpy as np
 
import argparse 

 
# Read-in params:

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False,
                        help='File with parameters (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False,
                        help='File to save generated data (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output

    
    
param_file, output_file = ParseArguments()
 

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)
 
 
w=params['w']
k=params['k']
alpha=params['alpha']
Theta = np.asarray(params['Theta'])
ThetaB =np.asarray(params['ThetaB'])


# TO DO: simulate x_1, ..., x_k, where x_i = (x_{i1}, ..., x_{iw})
# according to the description from the script, and save them in a .csv file
# the i-th row = x_i

def generate_sample(w, k, alpha, Theta, ThetaB):
    Z = np.random.choice([0, 1], size=k, p=[1 -alpha, alpha])
    X = np.zeros(shape=(k, w))

    for i in range(k):
        if Z[i]:
            r = np.zeros(shape=w)
            for j in range(w):
                r[j] = np.random.choice(np.arange(1, 5, step=1), p=Theta[:, j])
            X[i] = r
        else:
            X[i] = np.random.choice(np.arange(1, 5, step=1), size=w, p=ThetaB)
    return X, alpha


X, alpha = generate_sample(w, k, alpha, Theta, ThetaB)

# For example, assume that x_1, ..., x_k are collected in a matrix X
# (as a reminder: each x_{ij} is A, C, G, or T, which we identify with 1, 2, 3, 4)

# X = np.random.randint(4, size=(k, w)) + 1

# We need to save the matrix X and the value of alpha (k and w can be inferred from X)


gen_data = {    
    "alpha" : alpha,
    "X" : X.tolist()
    }



with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)
 


