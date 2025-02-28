import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm


from functions_Q1 import (default_pars, modes, smoothing, pca,
                           bootstrap, statistics, multiple_poisson_generator,
                           bin_spike_times, matrices, plots)

# Set the number of iterations to 2
title = ''
n_it = 2
lv_pc_corr = np.zeros((n_it, 2))

# Create a single 3D figure and axis to overlay both iterations
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Flag to plot the best-fitting plane only once
plane_plotted = False

for a in tqdm(range(n_it), desc="Iterating", unit="step", position=0, leave=True):
    ### 0: PREP
    pars = default_pars()
    seed = False
    bin_size = pars['bin_size']  # in ms
    num_bins = pars['T'] // bin_size

    # Initialize EVR arrays
    evr_boot_acc = np.zeros((num_bins, pars['n_it']))
    evr_boot_cumsum_acc = np.zeros((num_bins, pars['n_it']))

    evr_boot_mean = np.zeros((num_bins,))
    evr_boot_stdev = np.zeros((num_bins,))
    evr_boot_cumsum_mean = np.zeros((num_bins,))
    evr_boot_cumsum_stdev = np.zeros((num_bins,))
    evr_p = np.zeros((num_bins,))
    evr_cumsum_p = np.zeros((num_bins,))

    ### 1: SPIKE GENERATION
    lv_matrix, _ = modes(pars)
    acc_spike_trains = multiple_poisson_generator(pars, myseed=seed)

    ### 2: BINNING
    count_modes_avg, count_trials_avg = bin_spike_times(pars, num_bins, data=acc_spike_trains)

    ### 3: SMOOTHING
    count_trials_avg_smt = smoothing(pars, data=count_trials_avg)

    ### 4: SHUFFLING & PCA
    # Accumulator for shuffled mode counts
    count_modes_avg_shf_acc = np.zeros(count_modes_avg.shape)

    for i in range(pars['n_it']):
        # Bootstrap for trials and modes
        count_trials_avg_smt_shf = bootstrap(data=count_trials_avg_smt)
        count_modes_avg_smt_shf = bootstrap(data=count_modes_avg)
        count_modes_avg_shf_acc += count_modes_avg_smt_shf

        # PCA on the bootstrapped trial data
        _, evr_boot, evr_boot_cumsum = pca(data=count_trials_avg_smt_shf)
        evr_boot_acc[:, i] = evr_boot
        evr_boot_cumsum_acc[:, i] = evr_boot_cumsum

    # Average modes array from shuffling
    count_modes_avg_shf = count_modes_avg_shf_acc / pars['n_it']

    ### 5: PCA (un-shuffled)
    principal_components, evr_uns, evr_uns_cumsum = pca(data=count_trials_avg_smt)

    # Normalize count_modes_avg
    count_modes_avg_centered = count_modes_avg - np.mean(count_modes_avg)

    # STATISTICS
    lv_matrix_trans = lv_matrix.T
    lv_matrix_pre = lv_matrix_trans.reshape(lv_matrix.shape[1], num_bins, bin_size)
    lv_matrix_avg = lv_matrix_pre.mean(axis=2)

    r_lv1_inout, p_lv1_inout = pearsonr(lv_matrix_avg[0], count_modes_avg[0])
    r_lv2_inout, p_lv2_inout = pearsonr(lv_matrix_avg[1], count_modes_avg[1])
    r_lv, p_lv = pearsonr(count_modes_avg[0], count_modes_avg[1])

    evr_boot_mean, evr_boot_stdev, evr_p, pc_lv_corr, pc_lv_p = statistics(
        evr_boot_acc, evr_uns,
        PC_data=principal_components,
        LV_data=count_modes_avg_centered,
        n_lv=lv_matrix.shape[1]
    )

    evr_boot_cumsum_mean, evr_boot_cumsum_stdev, evr_cumsum_p, _, _ = statistics(
        evr_boot_cumsum_acc, evr_uns_cumsum, _,
        _,
        n_lv=lv_matrix.shape[1]
    )

    # Store the appropriate correlation diagonal
    main_diag = np.array([pc_lv_corr[0, 0], pc_lv_corr[1, 1]])
    anti_diag = np.array([pc_lv_corr[0, 1], pc_lv_corr[1, 0]])
    if np.any(main_diag < 0):
        diagonal = anti_diag
    else:
        diagonal = main_diag
    lv_pc_corr[a, :] = diagonal

    # Determine the number of PCs explaining more variance than shuffled
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

    x_counts = np.linspace(bin_size / 2, pars['T'] - bin_size / 2, num_bins)
    CVM, corr_matrix = matrices(data=count_trials_avg)

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



    # --- Prepare 3D scatter data ---
    # Extract PC1, PC2, and PC3
    pc1 = principal_components[0, :]
    pc2 = principal_components[1, :]
    pc3 = principal_components[2, :]
    pc_data = np.vstack((pc1, pc2, pc3)).T
    time_points = np.arange(len(pc1))

    # For the first iteration, compute and plot the best-fitting plane
    if not plane_plotted:
        U, s, Vt = np.linalg.svd(pc_data - np.mean(pc_data, axis=0), full_matrices=False)
        normal = Vt[-1]
        xx, yy = np.meshgrid(np.linspace(np.min(pc1), np.max(pc1), 10),
                             np.linspace(np.min(pc2), np.max(pc2), 10))
        d = -np.dot(np.mean(pc_data, axis=0), normal)
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        ax.plot_wireframe(xx, yy, zz, color='black', linewidth=1, alpha=0.5, rstride=9, cstride=9)
        plane_plotted = True

    # Set scatter opacity: full for iteration 0, 0.5 for iteration 1
    alpha_val = 1.0 if a == 0 else 0.5
    ax.scatter(pc1, pc2, pc3, c=time_points, cmap='viridis', alpha=alpha_val, s=80)

# --- Customize the overall 3D plot ---
ax.set_title('Projection of 80 Neurons on 3D PC-space', fontsize=20)
ax.set_xlabel('PC1', fontsize=20, labelpad=10)
ax.set_ylabel('PC2', fontsize=20, labelpad=10)
ax.set_zlabel('PC3', fontsize=20, labelpad=10)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)  # Zoom in on the z-axis (PC3 from -5 to 5)
ticks = [-10, 0, 10]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks([-5, 0, 5])
ax.tick_params(axis='x', pad=5, labelsize=16)
ax.tick_params(axis='y', pad=5, labelsize=16)
ax.tick_params(axis='z', pad=5, labelsize=16)
ax.view_init(elev=20, azim=45)

# Add a colorbar with a shrink factor to adjust its height relative to the axis
cbar = fig.colorbar(ax.collections[-1], ax=ax, shrink=0.5)
cbar.set_label('Time Points', fontsize=20)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.show()
