import numpy as np

def compute_log_likehood(X, alpha, Theta, ThetaB):
    k, w = X.shape
    log_motif = np.sum(np.log(Theta[X, np.arange(w)]), axis=1)
    log_bg = np.sum(np.log(ThetaB[X]), axis=1)

    return np.sum(np.log(alpha * np.exp(log_motif) + (1 - alpha) * np.exp(log_bg)))

def em_algorithm(X, alpha, Theta, ThetaB, max_iter = 1000, tol=1e-100, plot_ll=False):

    ll_differences = []
    ll_values = []

    X = np.array(X, dtype=int)
    k, w = X.shape
    Theta = np.array(Theta)
    ThetaB = np.array(ThetaB)
    maximal_t = 0
    for t in range(max_iter):

        # Expectation

        p_motif = np.prod(Theta[X, np.arange(w)], axis=1)
        p_bg = np.prod(ThetaB[X], axis=1)

        ll = np.sum(np.log(alpha * p_motif + (1 - alpha) * p_bg))

        Q1 = (alpha * p_motif) / sum([alpha * p_motif, (1-alpha) * p_bg])

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


        new_p_motif = np.prod(Theta[X, np.arange(w)], axis=1)
        new_p_bg = np.prod(ThetaB[X], axis=1)
        new_ll = np.sum(np.log(alpha * new_p_motif + (1 - alpha) * new_p_bg))

        maximal_t += 1

        ll_values.append(ll)
        ll_differences.append(abs(new_ll - ll))
        if not plot_ll:
            if abs(new_ll - ll) < tol:
                print(f"number of iterations: {maximal_t}")
                break
    Theta, ThetaB = Theta.round(20), ThetaB.round(20)

    if plot_ll:
        return Theta, ThetaB, ll_values, ll_differences

    return Theta, ThetaB


def var_dist(dist_1, dist_2):
    return 0.5 * np.sum(np.abs(dist_1 - dist_2))

def distance(dist_1, dist_2, dist_1_bg, dist_2_bg, plot=False):
    w = dist_1.shape[1]
    d_bg = var_dist(dist_1_bg, dist_2_bg)
    d_motif = 0.0
    for i in range(w):
        true = dist_1[:, i]
        est  = dist_2[:, i]
        d_motif += var_dist(true, est)

    if plot:
        return d_bg, d_motif / w, (d_bg + d_motif) / (w + 1)

    return (d_bg + d_motif) / (w + 1)