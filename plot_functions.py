import matplotlib.pyplot as plt
import json
from helper_functions import *
from motif_340146_336942_generate import generate_sample
import seaborn as sns


def visualize_k(param_file = "params_set1.json", ran = [50*k for k in range(1, 21)], av_count=30):

    results = []

    with open(param_file, 'r') as inputfile:
        params = json.load(inputfile)

    w = params['w']
    alpha = params['alpha']
    TrueTheta = np.asarray(params['Theta'])
    TrueThetaB = np.asarray(params['ThetaB'])

    for k in ran:
        d = []
        d_bg = []
        d_motif = []
        for i in range(av_count):
            np.random.seed(42+i)
            X, _ = generate_sample(w, k, alpha, TrueTheta, TrueThetaB)

            Theta = np.random.rand(4, w)
            Theta = Theta / Theta.sum(axis=0)

            ThetaB = np.random.rand(4)
            ThetaB = ThetaB / ThetaB.sum()

            Theta, ThetaB = em_algorithm(X, alpha, Theta, ThetaB, max_iter=1000, tol=1e-10)

            bg_d, motif_d, total_d = distance(TrueTheta, Theta, TrueThetaB, ThetaB)
            d_bg.append(bg_d)
            d_motif.append(motif_d)
            d.append(total_d)

        results.append({
            'alpha': alpha,
            'w': w,
            'k': k,
            'dtv_mean': np.mean(d),
            'dtv_std': np.std(d),
            'dtv_motif_mean': np.mean(d_motif),
            'dtv_motif_std': np.std(d_motif),
            'dtv_bg_mean': np.mean(d_bg),
            'dtv_bg_std': np.std(d_bg)
        })
    if not results:
        print("No results to plot.")
        return

    results.sort(key=lambda r: r['k'])

    k_vals = [r['k'] for r in results]
    dtv_means = [r['dtv_mean'] for r in results]
    dtv_stds = [r['dtv_std'] for r in results]
    motif_means = [r['dtv_motif_mean'] for r in results]
    bg_means = [r['dtv_bg_mean'] for r in results]
    alpha_val = results[0]['alpha']
    w_val = results[0]['w']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(k_vals, dtv_means, yerr=dtv_stds, fmt='-o', capsize=5, label='Total $d_{tv}$ (Avg)')

    ax.plot(k_vals, motif_means, '--s', label='Motif $d_{tv}$', alpha=0.8)
    ax.plot(k_vals, bg_means, '--^', label='Background $d_{tv}$', alpha=0.8)

    ax.set_title(f'Algorithm Performance vs. Sample Size (k)\n(α={alpha_val}, w={w_val}, {av_count} runs per point)', fontsize=16)
    ax.set_xlabel("Sample Size (k)", fontsize=16)
    ax.set_ylabel("Average Total Variation Distance ($d_{tv}$)", fontsize=16)
    ax.set_xticks(k_vals)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# visualize_k()


def visualize_alpha(param_file = "params_set1.json", ran = [10*a/100 for a in range(1, 10, 1)], av_count=15):
    results = []
    with open(param_file, 'r') as inputfile:
        params = json.load(inputfile)

    w = params['w']
    k = params['k']
    TrueTheta = np.asarray(params['Theta'])
    TrueThetaB = np.asarray(params['ThetaB'])

    for alpha in ran:
        d = []
        d_bg = []
        d_motif = []
        for i in range(av_count):
            np.random.seed(42+i)
            X, _ = generate_sample(w, k, alpha, TrueTheta, TrueThetaB)

            Theta = np.random.rand(4, w)
            Theta = Theta / Theta.sum(axis=0)

            ThetaB = np.random.rand(4)
            ThetaB = ThetaB / ThetaB.sum()

            Theta, ThetaB = em_algorithm(X, alpha, Theta, ThetaB, max_iter=1000, tol=1e-10)


            bg_d, motif_d, total_d = distance(TrueTheta, Theta, TrueThetaB, ThetaB)
            d_bg.append(bg_d)
            d_motif.append(motif_d)
            d.append(total_d)

        results.append({
            'alpha': alpha,
            'w': w,
            'k': k,
            'dtv_mean': np.mean(d),
            'dtv_std': np.std(d),
            'dtv_motif_mean': np.mean(d_motif),
            'dtv_motif_std': np.std(d_motif),
            'dtv_bg_mean': np.mean(d_bg),
            'dtv_bg_std': np.std(d_bg)
        })
    if not results:
        print("No results to plot.")
        return

    results.sort(key=lambda r: r['alpha'])

    alpha_vals = [r['alpha'] for r in results]
    dtv_means = [r['dtv_mean'] for r in results]
    dtv_stds = [r['dtv_std'] for r in results]
    motif_means = [r['dtv_motif_mean'] for r in results]
    bg_means = [r['dtv_bg_mean'] for r in results]
    k_val = results[0]['k']
    w_val = results[0]['w']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(alpha_vals, dtv_means, yerr=dtv_stds, fmt='-o', capsize=5, label='Total $d_{tv}$ (Avg)')

    ax.plot(alpha_vals, motif_means, '--s', label='Motif $d_{tv}$', alpha=0.8)
    ax.plot(alpha_vals, bg_means, '--^', label='Background $d_{tv}$', alpha=0.8)

    ax.set_title(f'Algorithm Performance vs. Motif Probability (α)\n(k={k_val}, w={w_val}, {av_count} runs per point)', fontsize=16)
    ax.set_xlabel("motif probability (α)", fontsize=16)
    ax.set_ylabel("Average Total Variation Distance ($d_{tv}$)", fontsize=16)
    ax.set_xticks(alpha_vals)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# visualize_alpha()


def visualize_w(param_files, av_count=15):

    results = []
    for param_file in param_files:
        with open(param_file, 'r') as f:
            params = json.load(f)

        w = params['w']
        alpha = params['alpha']
        k = params['k']
        TrueTheta = np.asarray(params['Theta'])
        TrueThetaB = np.asarray(params['ThetaB'])

        run_dtvs, run_dtvs_motif, run_dtvs_bg = [], [], []

        for i in range(av_count):
            np.random.seed(42+i)
            X, _ = generate_sample(w, k, alpha, TrueTheta, TrueThetaB)

            Theta_init = np.random.rand(4, w)
            Theta_init = Theta_init / Theta_init.sum(axis=0)

            ThetaB_init = np.random.rand(4)
            ThetaB_init = ThetaB_init / ThetaB_init.sum()

            Theta_est, ThetaB_est = em_algorithm(X, alpha, Theta_init, ThetaB_init, max_iter=1000, tol=1e-10)

            bg_d, motif_d, total_d = distance(TrueTheta, Theta_est, TrueThetaB, ThetaB_est)

            run_dtvs.append(total_d)
            run_dtvs_motif.append(motif_d)
            run_dtvs_bg.append(bg_d)

        results.append({
            'w': w,
            'k': k,
            'dtv_mean': np.mean(run_dtvs),
            'dtv_std': np.std(run_dtvs),
            'dtv_motif_mean': np.mean(run_dtvs_motif),
            'dtv_motif_std': np.std(run_dtvs_motif),
            'dtv_bg_mean': np.mean(run_dtvs_bg),
            'dtv_bg_std': np.std(run_dtvs_bg)
        })

    if not results:
        print("No results to plot.")
        return

    results.sort(key=lambda r: r['w'])

    w_vals = [r['w'] for r in results]
    dtv_means = [r['dtv_mean'] for r in results]
    dtv_stds = [r['dtv_std'] for r in results]
    motif_means = [r['dtv_motif_mean'] for r in results]
    bg_means = [r['dtv_bg_mean'] for r in results]
    k_val = results[0]['k']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(w_vals, dtv_means, yerr=dtv_stds, fmt='-o', capsize=5, label='Total $d_{tv}$ (Avg)')

    ax.plot(w_vals, motif_means, '--s', label='Motif $d_{tv}$', alpha=0.8)
    ax.plot(w_vals, bg_means, '--^', label='Background $d_{tv}$', alpha=0.8)

    ax.set_title(f'Algorithm Performance vs. Motif Length (w) (strong motif)  \n(k={k_val}, α={alpha}, {av_count} runs per point)', fontsize=16)
    ax.set_xlabel("Motif Length (w)", fontsize=16)
    ax.set_ylabel("Average Total Variation Distance ($d_{tv}$)", fontsize=16)
    ax.set_xticks(w_vals)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# param_files = [f"params_set_w{w_val}.json" for w_val in [3, 5, 8, 12]]
# visualize_w(param_files=param_files, av_count=15)

# param_files = [f"params_random_w{w_val}.json" for w_val in [3, 5, 8, 12]]
# visualize_w(param_files=param_files, av_count=15)

def visualize_motif_strength_boxplot(strength_param_files, strength_labels, av_count=15):
    """
    Visualizes algorithm performance against motif strength using boxplots.

    Each boxplot shows the distribution of the total variation distance error
    over multiple simulation runs for a given motif strength.
    """
    # This list will hold the lists of run results for each parameter file.
    # e.g., [[0.1, 0.12, 0.09, ...], [0.05, 0.04, 0.06, ...], ...]
    all_runs_data = []

    # Store k and alpha for the plot title (assuming they are consistent)
    k, alpha = None, None

    for param_file in strength_param_files:
        with open(param_file, 'r') as f:
            params = json.load(f)

        # Extract parameters. Store the last seen k and alpha for the title.
        w, k, alpha = params['w'], params['k'], params['alpha']
        TrueTheta = np.asarray(params['Theta'])
        TrueThetaB = np.asarray(params['ThetaB'])

        # This list will store errors for the CURRENT strength level
        run_dtvs = []
        for i in range(av_count):
            # Use a different seed for each run to ensure variability
            np.random.seed(42 + i)

            X, _ = generate_sample(w, k, alpha, TrueTheta, TrueThetaB)

            Theta_init = np.random.rand(4, w)
            Theta_init = Theta_init / Theta_init.sum(axis=0, keepdims=True)

            ThetaB_init = np.random.rand(4)
            ThetaB_init = ThetaB_init / ThetaB_init.sum()

            Theta_est, ThetaB_est = em_algorithm(X, alpha, Theta_init, ThetaB_init, max_iter=1000, tol=1e-10)

            _, _, total_error = distance(TrueTheta, Theta_est, TrueThetaB, ThetaB_est)
            run_dtvs.append(total_error)

        # KEY CHANGE: Append the entire list of raw error values for this strength level
        all_runs_data.append(run_dtvs)
        print(f"Completed simulations for file: {param_file}")

    if not all_runs_data:
        print("No results to plot.")
        return

    # --- Plotting Section ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # KEY CHANGE: Use seaborn's boxplot to visualize the distribution of errors
    sns.boxplot(data=all_runs_data, ax=ax, palette="viridis", width=0.6)

    # Update labels and title to be more descriptive of a distribution
    ax.set_ylabel("Distribution of Total Variation Distance ($d_{tv}$ Error)", fontsize=16)
    ax.set_xlabel("Motif Information Strength", fontsize=16)
    ax.set_xticklabels(strength_labels, rotation=0, fontsize=16)  # Rotate for readability

    # Check if k and alpha were found before adding to title
    if k is not None and alpha is not None:
        title = f'Algorithm Performance vs. Motif Strength\n(k={k}, α={alpha}, {av_count} runs per strength level)'
    else:
        title = f'Algorithm Performance vs. Motif Strength\n({av_count} runs per strength level)'
    ax.set_title(title, fontsize=16)

    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.show()

strength_files = [
        'params_strength_weak.json',
        'params_strength_medium.json',
        'params_strength_strong.json'
        ]

labels = [
        'Weak',
        'Medium',
        'Strong'
        ]

visualize_motif_strength_boxplot(strength_files, labels, av_count=20)

def visualize_ll(param_file = "params_set1.json", av_count=15):

    results = []

    with open(param_file, 'r') as inputfile:
        params = json.load(inputfile)

    w = params['w']
    alpha = params['alpha']
    k = params['k']
    TrueTheta = np.asarray(params['Theta'])
    TrueThetaB = np.asarray(params['ThetaB'])

    ll = []
    ll_d = []
    for i in range(av_count):
        np.random.seed(42+i)
        X, _ = generate_sample(w, k, alpha, TrueTheta, TrueThetaB)

        ThetaB = np.zeros(4)
        ThetaB[:(4 - 1)] = np.random.rand(4 - 1) / 4
        ThetaB[4 - 1] = 1 - np.sum(ThetaB)

        Theta = np.zeros((4, w))
        Theta[:(w), :] = np.random.random((3, w)) / w
        Theta[w, :] = 1 - np.sum(Theta, axis=0)

        _, _, ll_t, ll_dt = em_algorithm(X, alpha, Theta, ThetaB, max_iter=100, tol=1e-10, plot_ll=True)


        ll.append(ll_t)
        ll_d.append(ll_dt)
    ll = np.array(ll)
    ll_d = np.array(ll_d)


    ll_means = np.mean(ll, axis=0)
    ll_stds = np.std(ll, axis=0)
    ll_d_means = np.mean(ll_d, axis=0)
    ll_d_stds = np.std(ll_d, axis=0)


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations_show = [i for i in range(1, 101, 1)]
    print(len(ll_means))
    ax.errorbar(iterations_show, ll_means, fmt='-o', capsize=5, label='average log-likelihood value')

    ax.set_title(f'Mean Log-likelihood Values Across Iterations of EM Algorithm\n(α = {alpha}, k={k}, w={w}, {av_count} runs per point)', fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Average Log-likelihood Value (linear scale)", fontsize=16)
    # ax.set_xticks(iterations_show)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    # plt.yscale('log')
    plt.tight_layout()
    plt.show()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(iterations_show, ll_d_means, fmt='-o', capsize=5, label='average log-likelihood difference value')

    ax.set_title(
        f'Mean Log-likelihood Differences Across Iterations of EM Algorithm\n(α = {alpha}, k={k}, w={w}, {av_count} runs per point)', fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Average Log-likelihood Difference (log scale)", fontsize=16)
    # ax.set_xticks(iterations_show)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.legend(fontsize=14)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


visualize_ll()
