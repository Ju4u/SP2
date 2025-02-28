import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import pearsonr
from tqdm import tqdm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd


from functions_Q1 import (default_pars, modes, smoothing, pca,
                           bootstrap, statistics, multiple_poisson_generator,
                           bin_spike_times, matrices, plots)

n_it = 1
lv_pc_corr = np.zeros((n_it, 2))

for a in tqdm(range(n_it), desc="Iterating", unit="step", position=0, leave=True):
    ### 0: PREP
    pars = default_pars()
    seed = False
    bin_size = pars['bin_size']  # in ms
    num_bins = pars['T'] // bin_size

    # init evr arrays
    evr_boot_acc = np.zeros((num_bins, pars['n_it']))  # (80, 100)
    evr_boot_cumsum_acc = np.zeros((num_bins, pars['n_it']))  # (80, 100)

    evr_boot_mean = np.zeros((num_bins,))  # (80,)
    evr_boot_stdev = np.zeros((num_bins,))  # (80,)
    evr_boot_cumsum_mean = np.zeros((num_bins,))  # (80,)
    evr_boot_cumsum_stdev = np.zeros((num_bins,))  # (80,)
    evr_p = np.zeros((num_bins,))  # (80,)
    evr_cumsum_p = np.zeros((num_bins,))  # (80,)

    ### 1: SPIKE GENERATION
    lv_matrix, _ = modes(pars)
    acc_spike_trains = multiple_poisson_generator(pars, myseed=seed)

    ### 2: BINNING
    count_modes_avg, count_trials_avg = bin_spike_times(pars, num_bins, data=acc_spike_trains)

    ### 3: SMOOTHING
    count_trials_avg_smt = smoothing(pars, data=count_trials_avg)

    ### 4: SHUFFLING & PCA

    # init shuffled mode count array
    count_modes_avg_shf_acc = np.zeros(count_modes_avg.shape)

    for i in range(pars['n_it']):  # tqdm(range(pars['n_it']), desc="Analyzing", unit="step")
        # trials
        count_trials_avg_smt_shf = bootstrap(data=count_trials_avg_smt)

        # modes
        count_modes_avg_smt_shf = bootstrap(data=count_modes_avg)
        count_modes_avg_shf_acc += count_modes_avg_smt_shf

        # pca (evr_boot)
        _, evr_boot, evr_boot_cumsum = pca(data=count_trials_avg_smt_shf)
        evr_boot_acc[:, i] = evr_boot
        evr_boot_cumsum_acc[:, i] = evr_boot_cumsum

    # average modes array
    count_modes_avg_shf = count_modes_avg_shf_acc / pars['n_it']

    ### 5:  PCA (un-shuffled)
    principal_components, evr_uns, evr_uns_cumsum = pca(data=count_trials_avg_smt)


    # normalize count_modes_avg
    count_modes_avg_centered = count_modes_avg - np.mean(count_modes_avg)


    # STATISTICS
    # Bin input LVs
    lv_matrix_trans = lv_matrix.T
    lv_matrix_pre = lv_matrix_trans.reshape(lv_matrix.shape[1], num_bins, bin_size)
    lv_matrix_avg = lv_matrix_pre.mean(axis=2)

    # LVin --> LVout
    r_lv1_inout, p_lv1_inout = pearsonr(lv_matrix_avg[0], count_modes_avg[0])
    r_lv2_inout, p_lv2_inout = pearsonr(lv_matrix_avg[1], count_modes_avg[1])

    # LV1 --> LV2
    r_lv, p_lv = pearsonr(count_modes_avg[0], count_modes_avg[1])

    # LV1/2 --> PC1/2
    #unshuffled
    evr_boot_mean, evr_boot_stdev, evr_p, pc_lv_corr, pc_lv_p = statistics(evr_boot_acc, evr_uns,
                                                                           PC_data=principal_components,
                                                                           LV_data=count_modes_avg_centered,
                                                                           n_lv=lv_matrix.shape[1])


    #boot
    evr_boot_cumsum_mean, evr_boot_cumsum_stdev, evr_cumsum_p, _, _ = statistics(evr_boot_cumsum_acc, evr_uns_cumsum, _,
                                                                                 _,
                                                                                 n_lv=lv_matrix.shape[1])

    # #Append lv_pc_corr matrix
    # diagonals = [pc_lv_corr[0, 1], pc_lv_corr[1, 0]]
    # lv_pc_corr[a] = diagonals


    # Extract both diagonals
    main_diag = np.array([pc_lv_corr[0, 0], pc_lv_corr[1, 1]])  # Main diagonal [a, d]
    anti_diag = np.array([pc_lv_corr[0, 1], pc_lv_corr[1, 0]])  # Anti-diagonal [b, c]

    # Check which diagonal contains the negative value and select the other one
    if np.any(main_diag < 0):
        diagonal = anti_diag  # If main diagonal has a negative, take the anti-diagonal
    else:
        diagonal = main_diag  # Otherwise, take the main diagonal

    lv_pc_corr[a, :] = diagonal


    # Calculate number of PCs explaining more variance than shuffled
    npc = 0
    for i in range(len(evr_uns)):
        if evr_uns[i] > evr_boot_mean[i]:
            npc += 1
        else:
            break
    int_uns = evr_uns_cumsum[npc]
    int_boot = evr_boot_cumsum_mean[npc]

    # Calculate confidence thresholds
    evr_boot_plus = evr_boot_mean + evr_boot_stdev
    evr_boot_minus = evr_boot_mean - evr_boot_stdev

    evr_boot_cumsum_plus = evr_boot_cumsum_mean + evr_boot_cumsum_stdev
    evr_boot_cumsum_minus = evr_boot_cumsum_mean - evr_boot_cumsum_stdev

    ### PLOTTING
    x_counts = np.linspace(bin_size / 2, pars['T'] - bin_size / 2, num_bins)
    CVM, corr_matrix = matrices(data=count_trials_avg)
    title = (f'No smoothing,\n'
             f'r(LVin-->LVout)1 = {r_lv1_inout:.3f}\n'
             f'r(LVin-->LVout)2 = {r_lv2_inout:.3f},\n'
             f'r(LV1-->LV2)out = {r_lv:.3f}')

    # plots(pars, range_t=pars['range_t'],
    #       spike_trains=acc_spike_trains,
    #       count_modes_avg=count_modes_avg,
    #       x_counts=x_counts,
    #       CVM=CVM,
    #       corr_matrix=corr_matrix,
    #       lv_matrix=lv_matrix,
    #       principal_components=principal_components,
    #       count_modes_boot_avg=count_modes_avg_shf,
    #       evr_uns=evr_uns,
    #       evr_uns_cumsum=evr_uns_cumsum,
    #       evr_boot=evr_boot_mean,
    #       evr_boot_cumsum=evr_boot_cumsum_mean,
    #       evr_boot_plus=evr_boot_plus,
    #       evr_boot_minus=evr_boot_minus,
    #       evr_boot_cumsum_plus=evr_boot_cumsum_plus,
    #       evr_boot_cumsum_minus=evr_boot_cumsum_minus,
    #       n_it=pars['n_it'],
    #       npc=npc,
    #       int_uns=int_uns,
    #       int_boot=int_boot,
    #       title=title,
    #       p=evr_p,
    #       p_cumsum=evr_cumsum_p,
    #       r_lv=r_lv,
    #       p_lv=p_lv,
    #       lv_data=lv_matrix_avg,
    #       pc_lv_corr=pc_lv_corr
    #       )


# corrmean_pc1 = np.mean(lv_pc_corr[:, 0])
# corrmean_pc2 = np.mean(lv_pc_corr[:, 1])
# np.save('lv_pc_corr_04.npy', lv_pc_corr)

# # exclude outliers
# # lv_pc_corr = lv_pc_corr[(lv_pc_corr >= 0).all(axis=1)]
#
#
# # Set a clean style with a color palette
# sns.set_theme(style="whitegrid", palette="pastel")
#
# plt.figure(figsize=(10, 5))
#
# # Plot first column with purple color, thinner bars, and lower alpha for heatmap effect
# sns.histplot(lv_pc_corr[:, 0], kde=True, binwidth=0.01, color='purple', label=f"PC1, mean = {corrmean_pc1:.3f}", alpha=0.1)
#
# # Plot second column with blue color, thinner bars, and lower alpha for heatmap effect
# sns.histplot(lv_pc_corr[:, 1], kde=True, binwidth=0.01, color='blue', label=f"PC2, mean = {corrmean_pc2:.3f}", alpha=0.1)
#
# # Adjusting labels and title
# plt.xlabel("r(PC-LV)", fontsize=20)
# plt.ylabel("Frequency", fontsize=20)
# plt.title(f"Distribution of PC_mins (r=0.41, N={lv_pc_corr.shape[0]})", fontsize=30, fontweight='bold')
#
# # Customizing legend
# plt.legend(title="Principal Components", loc="upper left", fontsize=20)
#
# # Make the grid less intrusive and improve ticks
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlim(0.0, 1)
#
# # Show the plot
# plt.tight_layout()
# plt.show()



