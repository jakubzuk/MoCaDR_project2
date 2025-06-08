import numpy as np
import json

with open("generated_data.json", 'r') as inputfile:
    params = json.load(inputfile)

alpha = params['alpha']
X = params['X']
w = 3
k = 10
# Theta = np.array([[0.375, 0.1, 0.14285714285714285], [0.125, 0.2, 0.2857142857142857],
#           [0.25, 0.3, 0.14285714285714285], [0.25, 0.4, 0.42857142857142855]])
# ThetaB = np.array([0.25, 0.25, 0.25, 0.25])
# X = np.random.randint(0, 4, (k, w))
# print(X)
# print(Theta[X, np.arange(w)])

Theta = np.random.uniform(0, 1, 4*w)
Theta = Theta.reshape(4, w)
Theta = Theta / Theta.sum(axis=0)
print(Theta)

ThetaB = np.random.uniform(0,1, 4)
ThetaB = ThetaB / ThetaB.sum()
print(ThetaB)

def compute_log_likehood(X, alpha, Theta, ThetaB):
    k, w = X.shape
    log_motif = np.sum(np.log(Theta[X, np.arange(w)]), axis=1)
    log_bg = np.sum(np.log(ThetaB[X]), axis=1)

    return np.sum(np.log(alpha * np.exp(log_motif) + (1 - alpha) * np.exp(log_bg)))

def em_algorithm(X, alpha, Theta, ThetaB, max_iter = 1000, tol=1e-10):

    X = np.array(X, dtype=int)
    k, w = X.shape
    Theta = np.array(Theta)
    ThetaB = np.array(ThetaB)
    maximal_t = 0
    for t in range(max_iter):

        # Expectation

        # log_motif = np.sum(np.log(Theta[X, np.arange(w)]), axis=1)
        # log_bg = np.sum(np.log(ThetaB[X]), axis=1)

        # ll = np.sum(np.log(alpha * np.exp(log_motif) + (1 - alpha) * np.exp(log_bg)))

        # p_motif = np.exp(log_motif)
        # p_bg = np.exp(log_bg)

        p_motif = np.prod(Theta[X, np.arange(w)], axis=1)
        p_bg = np.prod(ThetaB[X], axis=1)

        ll = np.sum(np.log(alpha * p_motif + (1 - alpha) * p_bg))

        Q1 = (alpha * p_motif) / sum([alpha * p_motif, (1-alpha) * p_bg])

        # ll = compute_log_likehood(X, alpha, Theta, ThetaB)

        # Maximization

        for j in range(w):
            counts_motif = np.zeros(4)
            for i in range(k):
                counts_motif[X[i, j]] += Q1[i]
            Theta[:, j] = counts_motif / Q1.sum()

        counts_bg = np.zeros(4)
        for i in range(k):
            counts_bg += (1 - Q1[i]) * np.bincount(X[i], minlength=4)
        ThetaB = counts_bg / counts_bg.sum()

        # new_ll = compute_log_likehood(X, alpha, Theta, ThetaB)

        new_p_motif = np.prod(Theta[X, np.arange(w)], axis=1)
        new_p_bg = np.prod(ThetaB[X], axis=1)
        new_ll = np.sum(np.log(alpha * new_p_motif + (1 - alpha) * new_p_bg))

        maximal_t += 1

        if abs(new_ll - ll) < tol:
            print(f"number of iterations: {maximal_t}")
            break
    Theta, ThetaB = Theta.round(20), ThetaB.round(20)
    return Theta, ThetaB

theta, thetaB = em_algorithm(X, alpha, Theta, ThetaB)

# print(theta)
# print(thetaB)
