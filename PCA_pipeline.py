import numpy as np
import time
from tqdm import tqdm

from functions_PCA_pipeline import (default_pars, modes, smoothing, pca,
                           bootstrap, statistics, multiple_poisson_generator,
                           bin_spike_times, matrices, plots)


### 0: PREP
start = time.perf_counter()
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


title = 'bin --> smooth --> shuffle'
### 2: BINNING
count_modes_avg, count_trials_avg = bin_spike_times(pars, num_bins, data=acc_spike_trains)

### 3: SMOOTHING
count_trials_avg_smt = smoothing(pars, data=count_trials_avg)

### 4: SHUFFLING & PCA

# init shuffled mode count array
count_modes_avg_shf_acc = np.zeros(count_modes_avg.shape)

for i in tqdm(range(pars['n_it']), desc="Analyzing", unit="step"):
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

# generate statistics arrays
#uns
evr_boot_mean, evr_boot_stdev, evr_p, pc_lv_corr, pc_lv_p = statistics(evr_boot_acc, evr_uns,
                                                                       PC_data=principal_components,
                                                                       LV_data=count_modes_avg_centered,
                                                                       n_lv=lv_matrix.shape[1])
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
      p_cumsum=evr_cumsum_p
      )

end = time.perf_counter()
print(f"Finished! Elapsed time: {end - start:.2f} s")