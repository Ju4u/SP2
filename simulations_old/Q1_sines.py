import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import pearsonr
from tqdm import tqdm
import seaborn as sns

from functions_Q1 import (default_pars, modes, smoothing, pca,
                           bootstrap, statistics, multiple_poisson_generator,
                           bin_spike_times, matrices, plots)
n_it = 1
pc_min = np.zeros((n_it, 2))

for a in tqdm(range(n_it), desc="Iterating", unit="step", position=0, leave=True):
    ### 0: PREP
    #start = time.perf_counter()
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

    # pearson = np.zeros(4)

    ### 1: SPIKE GENERATION
    lv_matrix, wgt_matrix = modes(pars)
    acc_spike_trains = multiple_poisson_generator(pars, myseed=seed)


    title = 'bin --> shuffle (no smoothing!)'
    ### 2: BINNING
    count_modes_avg, count_trials_avg = bin_spike_times(pars, num_bins, data=acc_spike_trains)


    ### 3: SMOOTHING
    count_trials_avg_smt = smoothing(pars, data=count_trials_avg)

    ### 4: SHUFFLING & PCA

    # init shuffled mode count array
    count_modes_avg_shf_acc = np.zeros(count_modes_avg.shape)

    for i in range(pars['n_it']): #tqdm(range(pars['n_it']), desc="Analyzing", unit="step")
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

    # #store PC1 and PC2 minima indices
    # pc1_argmin = np.argmin(principal_components[0, :])
    # pc2_argmin = np.argmin(principal_components[1, :])
    # pc_min[a] = [pc1_argmin, pc2_argmin]

    # normalize count_modes_avg
    count_modes_avg_centered = count_modes_avg - np.mean(count_modes_avg)

    # generate statistics arrays
    #uns
    evr_boot_mean, evr_boot_stdev, evr_p, pc_lv_corr, pc_lv_p = statistics(evr_boot_acc, evr_uns,
                                                                           PC_data=principal_components,
                                                                           LV_data=count_modes_avg_centered,
                                                                           n_lv=lv_matrix.shape[1])

    #Calc pearsons comparing LV sets
    r_lv, p_lv = pearsonr(count_modes_avg[0], count_modes_avg[1])

    #Bin input LVs
    lv_matrix_trans = lv_matrix.T
    lv_matrix_pre = lv_matrix_trans.reshape(lv_matrix.shape[1], num_bins, bin_size)
    lv_matrix_avg = lv_matrix_pre.mean(axis=2)

    #Calc pearsons comparing input with output LVs
    r_lv1_inout, p_lv1_inout = pearsonr(lv_matrix_avg[0], count_modes_avg[0])
    r_lv2_inout, p_lv2_inout = pearsonr(lv_matrix_avg[1], count_modes_avg[1])

    # Initialize empty lists to store the results
    # r_lv = np.zeros(3)
    # p_lv = np.zeros(3)
    #
    # # Loop over the pairs of columns (i, j) and calculate Pearson correlation
    # for i, j in [(0, 1), (1, 2), (2, 0)]:
    #     corr, p_value = pearsonr(count_modes_avg[i], count_modes_avg[j])
    #     r_lv[i] = corr
    #     p_lv[i] = p_value  # Convert p-value to the required scale





    #cumsum
    evr_boot_cumsum_mean, evr_boot_cumsum_stdev, evr_cumsum_p, _, _ = statistics(evr_boot_cumsum_acc, evr_uns_cumsum, _, _,
                                                                                 n_lv=lv_matrix.shape[1])


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

    plots(pars, range_t=pars['range_t'],
          spike_trains=acc_spike_trains,
          count_modes_avg=count_modes_avg,
          x_counts=x_counts,
          CVM=CVM,
          corr_matrix=corr_matrix,
          lv_matrix=lv_matrix,
          principal_components=principal_components,
          count_modes_boot_avg=count_modes_avg_shf,
          evr_uns=evr_uns,
          evr_uns_cumsum=evr_uns_cumsum,
          evr_boot=evr_boot_mean,
          evr_boot_cumsum=evr_boot_cumsum_mean,
          evr_boot_plus=evr_boot_plus,
          evr_boot_minus=evr_boot_minus,
          evr_boot_cumsum_plus=evr_boot_cumsum_plus,
          evr_boot_cumsum_minus=evr_boot_cumsum_minus,
          n_it=pars['n_it'],
          npc=npc,
          int_uns=int_uns,
          int_boot=int_boot,
          title=title,
          p=evr_p,
          p_cumsum=evr_cumsum_p,
          r_lv=r_lv,
          p_lv=p_lv,
          lv_data=lv_matrix_avg,
          pc_lv_corr=pc_lv_corr
          )

    #end = time.perf_counter()
    #print(f"Finished! Elapsed time: {end - start:.2f} s")

# # Create a figure
# plt.figure(figsize=(10, 5))
#
# # Plot first column
# sns.histplot(pc_min[:, 0], kde=True, bins=20, color='blue', label="PC1")
#
# # Plot second column
# sns.histplot(pc_min[:, 1], kde=True, bins=20, color='red', label="PC2")
#
# # Labels and legend
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.title("Distribution PC_mins")
# plt.legend()
# plt.show()

