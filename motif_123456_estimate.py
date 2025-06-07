import json 

import numpy as np
 
import argparse 

 
# Read-in params

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False,
                        help='File with input data (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False,
                        help='File where the estimated parameters will be saved (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False,
                        help='Should alpha be estimated or not? (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha

    
input_file, output_file, estimate_alpha = ParseArguments()
 


with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)
 
 
 
alpha=data['alpha']
X= np.asarray(data['X'])
k,w = X.shape


# TO DO: MAIN PART: Estimate Theta and ThetaB using EM and save to output_file
# Theta0 = vector of length w
# Theta = matrix of size d by w = 4 by w
# example is random (placeholder)


ThetaB=np.zeros(4)
ThetaB[:(4-1)]=np.random.rand(4-1)/4
ThetaB[4-1]=1-np.sum(ThetaB)

Theta = np.zeros((4,w))
Theta[:(w),:]=np.random.random((3,w))/w
Theta[w,:]=1-np.sum(Theta,axis=0)
# BONUS TASK: if estimate_alpha == "yes", then
# alpha must also be estimated (ignore the value provided in input_file)

estimated_params = {
    "alpha" : alpha,            # we "copy" this alpha — it was not estimated
    "Theta" : Theta.tolist(),   # estimated
    "ThetaB" : ThetaB.tolist()  # estimated
    }


with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)
    
    
    
