import matplotlib.pyplot as plt
import json
from helper_functions import *
from motif_340146_336942_generate import generate_sample


# def visualize_k(param_file = "params_set1.json", ran = [10*k for k in range(1, 51)], av_count=15):
#     with open(param_file, 'r') as inputfile:
#         params = json.load(inputfile)
#
#     w = params['w']
#     alpha = params['alpha']
#     TrueTheta = np.asarray(params['Theta'])
#     TrueThetaB = np.asarray(params['ThetaB'])
#
#     dtv = []
#     dtv_bg = []
#     dtv_motif = []
#     for k in ran:
#         d = 0
#         d_bg = 0
#         d_motif = 0
#         for i in range(av_count):
#             np.random.seed(42+i)
#             X, _ = generate_sample(w, k, alpha, TrueTheta, TrueThetaB)
#
#             ThetaB = np.zeros(4)
#             ThetaB[:(4 - 1)] = np.random.rand(4 - 1) / 4
#             ThetaB[4 - 1] = 1 - np.sum(ThetaB)
#
#             Theta = np.zeros((4, w))
#             Theta[:(w), :] = np.random.random((3, w)) / w
#             Theta[w, :] = 1 - np.sum(Theta, axis=0)
#
#             Theta, ThetaB = em_algorithm(X, alpha, Theta, ThetaB, max_iter=1000, tol=1e-10)
#
#             c = distance(TrueTheta, Theta, TrueThetaB, ThetaB)
#             d_bg += c[0]
#             d_motif += c[1]
#             d += c[2]
#
#         dtv.append(d/av_count)
#         dtv_bg.append(d_bg / av_count)
#         dtv_motif.append(d_motif / av_count)
#         print(k)
#
#     plt.scatter(ran, dtv)
#     plt.grid()
#     plt.title("Mean total variation distance vs. sample size")
#     plt.xlabel("Sample size")
#     plt.ylabel("$d_{tv}$")
#     plt.ylim(0, 0.5)
#     plt.show()
#
#     plt.scatter(ran, dtv_bg)
#     plt.grid()
#     plt.title("Mean total variation distance (background) vs. sample size")
#     plt.xlabel("Sample size")
#     plt.ylabel("$d_{tv}$")
#     plt.ylim(0, 0.5)
#     plt.show()
#
#     plt.scatter(ran, dtv_motif)
#     plt.grid()
#     plt.title("Mean total variation distance (motif) vs. sample size")
#     plt.xlabel("Sample size")
#     plt.ylabel("$d_{tv}$")
#     plt.ylim(0, 0.5)
#     plt.show()

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

            ThetaB = np.zeros(4)
            ThetaB[:(4 - 1)] = np.random.rand(4 - 1) / 4
            ThetaB[4 - 1] = 1 - np.sum(ThetaB)

            Theta = np.zeros((4, w))
            Theta[:(w), :] = np.random.random((3, w)) / w
            Theta[w, :] = 1 - np.sum(Theta, axis=0)

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

    ax.set_title(f'Algorithm Performance vs. Sample Size (k)\n(α={alpha_val}, w={w_val}, {av_count} runs per point)')
    ax.set_xlabel("Sample Size (k)")
    ax.set_ylabel("Average Total Variation Distance ($d_{tv}$)")
    ax.set_xticks(k_vals)
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

            ThetaB = np.zeros(4)
            ThetaB[:(4 - 1)] = np.random.rand(4 - 1) / 4
            ThetaB[4 - 1] = 1 - np.sum(ThetaB)

            Theta = np.zeros((4, w))
            Theta[:(w), :] = np.random.random((3, w)) / w
            Theta[w, :] = 1 - np.sum(Theta, axis=0)

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

    ax.set_title(f'Algorithm Performance vs. Motif Probability (α)\n(k={k_val}, w={w_val}, {av_count} runs per point)')
    ax.set_xlabel("motif probability (α)")
    ax.set_ylabel("Average Total Variation Distance ($d_{tv}$)")
    ax.set_xticks(alpha_vals)
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

    # --- Step 2: Create the plot ---
    if not results:
        print("No results to plot.")
        return

    # Sort results by w for clean plotting
    results.sort(key=lambda r: r['w'])

    w_vals = [r['w'] for r in results]
    dtv_means = [r['dtv_mean'] for r in results]
    dtv_stds = [r['dtv_std'] for r in results]
    motif_means = [r['dtv_motif_mean'] for r in results]
    bg_means = [r['dtv_bg_mean'] for r in results]
    k_val = results[0]['k']  # Assuming k is constant across files for the title

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot total d_tv with error bars
    ax.errorbar(w_vals, dtv_means, yerr=dtv_stds, fmt='-o', capsize=5, label='Total $d_{tv}$ (Avg)')

    # Plot components
    ax.plot(w_vals, motif_means, '--s', label='Motif $d_{tv}$', alpha=0.8)
    ax.plot(w_vals, bg_means, '--^', label='Background $d_{tv}$', alpha=0.8)

    ax.set_title(f'Algorithm Performance vs. Motif Length (w)\n(k={k_val}, α={alpha}, {av_count} runs per point)')
    ax.set_xlabel("Motif Length (w)")
    ax.set_ylabel("Average Total Variation Distance ($d_{tv}$)")
    ax.set_xticks(w_vals)  # Ensure all w values are shown as ticks
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# param_files = [f"params_set_w{w_val}.json" for w_val in [3, 5, 8, 12]]
# visualize_w(param_files=param_files, av_count=15)

def visualize_motif_strength(strength_param_files, strength_labels, av_count=15):
    results = []
    for param_file in strength_param_files:
        with open(param_file, 'r') as f:
            params = json.load(f)


        w, k, alpha = params['w'], params['k'], params['alpha']
        TrueTheta = np.asarray(params['Theta'])
        TrueThetaB = np.asarray(params['ThetaB'])

        TrueThetaB_reshaped = TrueThetaB.reshape(-1, 1)
        motif_strength = np.mean(var_dist(TrueTheta, TrueThetaB_reshaped))

        run_dtvs = []
        for i in range(av_count):

            np.random.seed(42 + i)
            X, _ = generate_sample(w, k, alpha, TrueTheta, TrueThetaB)

            Theta_init = np.random.rand(4, w)
            Theta_init = Theta_init / Theta_init.sum(axis=0, keepdims=True)

            ThetaB_init = np.random.rand(4)
            ThetaB_init = ThetaB_init / ThetaB_init.sum()

            Theta_est, ThetaB_est = em_algorithm(X, alpha, Theta_init, ThetaB_init, max_iter=1000, tol=1e-10)

            _, _, total_error = distance(TrueTheta, Theta_est, TrueThetaB, ThetaB_est)
            run_dtvs.append(total_error)

        results.append({
            'strength': motif_strength,
            'error_mean': np.mean(run_dtvs),
            'error_std': np.std(run_dtvs)
        })
        print(f"Completed simulations for file: {param_file}")

    if not results:
        print("No results to plot.")
        return

    error_means = [r['error_mean'] for r in results]
    error_stds = [r['error_std'] for r in results]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = np.arange(len(strength_labels))
    bars = ax.bar(x_pos, error_means, yerr=error_stds, align='center',
                  alpha=0.7, ecolor='black', capsize=10)

    ax.set_ylabel("Average Total Variation Distance ($d_{tv}$ Error)")
    ax.set_xlabel("Motif Information Strength")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strength_labels)
    ax.set_title(f'Algorithm Performance vs. Motif Strength\n(k={k}, α={alpha}, {av_count} runs per strength level)')
    ax.yaxis.grid(True)

    # Add the numerical strength value below each label for more information
    # for i, res in enumerate(results):
    #     ax.text(i, -0.05, f"(Strength: {res['strength']:.2f})",
    #             ha='center', transform=ax.get_xaxis_transform(), color='gray')

    plt.tight_layout(pad=2)
    plt.show()

# param_files_for_strength_test = [
#     "params_strength_weak.json",
#     "params_strength_medium.json",
#     "params_strength_strong.json"
# ]
#
# # Labels for the x-axis of the bar chart
# bar_labels = ["Weak", "Medium", "Strong"]
#
# visualize_motif_strength(
#     strength_param_files=param_files_for_strength_test,
#     strength_labels=bar_labels,
#     av_count=15
# )