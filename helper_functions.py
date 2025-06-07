import numpy as np

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