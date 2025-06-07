import numpy as np

w = 3
alpha = 0.5
k = 10
Theta = np.array([[0.375, 0.1, 0.14285714285714285], [0.125, 0.2, 0.2857142857142857],
          [0.25, 0.3, 0.14285714285714285], [0.25, 0.4, 0.42857142857142855]])
ThetaB = np.array([0.25, 0.25, 0.25, 0.25])
X = np.random.randint(0, 4, (k, w))
print(X)
print(Theta[X, np.arange(w)])

def compute_log_likehood(X, alpha, Theta, ThetaB):
    k, w = X.shape
    log_motif = np.sum(np.log(Theta[X, np.arange(w)]), axis=1)
    log_bg = np.sum(np.log(ThetaB[X]), axis=1)

    return np.sum(np.log(alpha * np.exp(log_motif) + (1 - alpha) * np.exp(log_bg)))

def em_algorithm(X, alpha, Theta, ThetaB, max_iter = 1000, tol=1e-5):
    X = np.array(X, dtype=int)
    Theta = np.array(Theta)
    ThetaB = np.array(ThetaB)
    for t in range(max_iter):

        # Expectation

        log_motif = np.sum(np.log(Theta[X, np.arange(w)]), axis=1)
        log_bg = np.sum(np.log(ThetaB[X]), axis=1)
        
        p_motif = np.exp(log_motif)
        p_bg = np.exp(log_bg)

        gamma = (alpha * p_motif) / (alpha * p_motif + (1 - alpha) * p_bg)

        ll = compute_log_likehood(X, alpha, Theta, ThetaB)

        # Maximization

        for j in range(w):
            counts_motif = np.zeros(4)
            for i in range(k):
                counts_motif[X[i, j]] += gamma[i]
            Theta[:, j] = counts_motif / gamma.sum()

        counts_bg = np.zeros(4)
        for i in range(k):
            counts_bg += (1 - gamma[i]) * np.bincount(X[i], minlength=4)
        ThetaB = counts_bg / counts_bg.sum()

        new_ll = compute_log_likehood(X, alpha, Theta, ThetaB)

        if abs(new_ll - ll) < tol:
            break

    return Theta, ThetaB 
    
theta, thetaB = em_algorithm(X, alpha, Theta, ThetaB)

print(compute_log_likehood(X, alpha, Theta, ThetaB))
print(theta)
print(thetaB)
