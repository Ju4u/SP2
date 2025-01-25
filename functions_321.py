# Imports
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
from scipy.interpolate import make_interp_spline as mis
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


def default_pars(**kwargs):
    pars = {}

    #simulation parameters
    pars['T'] = 8000  # Total duration of simulation [ms]
    pars['dt'] = 1  # Simulation time step [ms]
    pars['init_rate_rmp'] = 10
    pars['init_rate'] = 20
    pars['fin_rate'] = 30
    pars['n_periods'] = 1
    pars['n_neurons'] = 160
    pars['bin_size'] = 100
    pars['trials'] = 100
    pars['noise_amp'] = 0
    pars['n_it'] = 100
    pars['window_size'] = 5

    #Weights for LVs
    pars['w11'], pars['w12'], pars['w13'], pars['w14'] = 1, 0, 0, 0  #train 1-40 (steady state)
    pars['w21'], pars['w22'], pars['w23'], pars['w24'] = 0, 1, 0, 0  #train 41-80 (ramp)
    pars['w31'], pars['w32'], pars['w33'], pars['w34'] = 0, 0, 1, 0  #train 81-120 (sine)
    pars['w41'], pars['w42'], pars['w43'], pars['w44'] = 0, 0, 0, 1  #train 121-160 (shifted sine)

    #external parameters if any
    for k in kwargs:
        pars[k] = kwargs[k]

    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # Vector of discretized
    # time points [ms]
    return pars


def modes(pars):
    #LV matrix
    #1 steady state
    rates_sdy = np.full(pars['T'], pars['init_rate'])

    #3 sine
    x = np.linspace(0, 2 * pars['n_periods'] * np.pi, pars['T'])
    rates_sin = 10 * np.sin(x) + pars['init_rate']

    #2 ramp
    rates_rmp = np.linspace(pars['init_rate_rmp'], pars['fin_rate'], pars['T'])

    #3 shifted sine
    rates_ssine = 10 * np.sin(x + (np.pi / 2)) + pars['init_rate']

    #stacking
    lv_matrix = np.stack([rates_sdy, rates_rmp, rates_sin, rates_ssine], axis=1)

    #Weight matrix
    n_lv = lv_matrix.shape[1]
    nrns_per_mode = pars['n_neurons'] // n_lv
    wgt_matrix = np.zeros((n_lv, pars['n_neurons']))

    for i in range(nrns_per_mode):
        wgt_matrix[0, i] = pars['w11']
        wgt_matrix[1, i] = pars['w12']
        wgt_matrix[2, i] = pars['w13']
        wgt_matrix[3, i] = pars['w14']

        wgt_matrix[0, i + nrns_per_mode] = pars['w21']
        wgt_matrix[1, i + nrns_per_mode] = pars['w22']
        wgt_matrix[2, i + nrns_per_mode] = pars['w23']
        wgt_matrix[3, i + nrns_per_mode] = pars['w24']

        wgt_matrix[0, i + (2 * nrns_per_mode)] = pars['w31']
        wgt_matrix[1, i + (2 * nrns_per_mode)] = pars['w32']
        wgt_matrix[2, i + (2 * nrns_per_mode)] = pars['w33']
        wgt_matrix[3, i + (2 * nrns_per_mode)] = pars['w34']

        wgt_matrix[0, i + (3 * nrns_per_mode)] = pars['w41']
        wgt_matrix[1, i + (3 * nrns_per_mode)] = pars['w42']
        wgt_matrix[2, i + (3 * nrns_per_mode)] = pars['w43']
        wgt_matrix[3, i + (3 * nrns_per_mode)] = pars['w44']

    return lv_matrix, wgt_matrix


def smoothing(pars, data):

    data_smoothed = np.zeros(data.shape)
    window = np.ones(pars['window_size']) / pars['window_size']

    for i in range(data.shape[0]):
        #smoothe by average [window_size] values
        smoothed = np.convolve(data[i], window, mode='same')
        #trim to remove edge artifacts
        trim_length = pars['window_size'] // 2
        trimmed_smoothed = smoothed[trim_length:-trim_length]
        #pad to restore original dims (pad w/ original values)
        padded_smoothed = np.pad(trimmed_smoothed, (trim_length, trim_length), mode='edge')
        data_smoothed[i] = padded_smoothed

    return data_smoothed


def pca(data):
    # Transpose the data
    data_transposed = data.T

    # Standardize the transposed data
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data_transposed)

    # Apply PCA
    pca = PCA()  #n_components=lv_matrix.shape[1]
    principal_components_pre = pca.fit_transform(data_std)

    # Transpose back to get the desired shape (3, 8000)
    principal_components = principal_components_pre.T

    #Create Explained Variance ratio array
    evr = pca.explained_variance_ratio_
    evr_cumsum = np.cumsum(evr)

    return principal_components, evr, evr_cumsum


def bootstrap(data):
    '''
    shuffles columns to destroy temporal structure
    '''

    #init shuffle array
    data_shuffled = np.zeros_like(data)

    #append shuffled version
    for i in range(data.shape[0]):
        shuffled_indices = np.random.permutation(data.shape[1])
        data_shuffled[i] = data[i, shuffled_indices]

    return data_shuffled


def statistics(evr_data, datapoint):
    stdev = np.std(evr_data, axis=1)
    mean = np.mean(evr_data, axis=1)

    d = norm(loc=mean, scale=stdev)
    alpha = 1 - d.cdf(datapoint)

    return mean, stdev, alpha


def multiple_poisson_generator(pars, myseed=False):
    """
  Generates poisson trains

  Args:
    pars       : parameter dictionary
    myseed     : random seed. int or boolean

  Returns:
    mother_spike_trains : spike train matrix, ith row represents whether
                      there is a spike in ith spike train over time
                      (1 if spike, 0 otherwise)
  """

    # Retrieve simulation parameters
    dt, range_t = pars['dt'], pars['range_t']
    lv_matrix, wgt_matrix = modes(pars)

    Lt = range_t.size
    n_trains = wgt_matrix.shape[1]

    # set random seed
    if myseed:
        np.random.seed(seed=myseed)
    else:
        np.random.seed()

    #initialize spike trains matrix
    acc_spike_trains = np.zeros((n_trains * pars['trials'], Lt))

    # Compute rates for all neurons by combining contributions from all latent variables
    rates_matrix = np.dot(wgt_matrix.T, lv_matrix.T)  # Shape: (n_trains, Lt)

    for neuron in tqdm(range(n_trains), desc='Spiking', unit='step'):
        for trial in range(pars['trials']):
            #generate and #normalize noise
            noise = np.random.normal(loc=0, scale=pars['noise_amp'], size=pars['T'])
            #noise -= np.mean(noise)

            #generate uniformly distributed random variables + noise
            u_rand = np.random.rand(Lt)

            #generate poisson train for this trial and mode
            spike_train = (u_rand < (rates_matrix[neuron] + noise) * (dt / 1000.)).astype(float)

            acc_spike_trains[neuron * pars['trials'] + trial] = spike_train

    return acc_spike_trains


def bin_spike_times(pars, num_bins, data):
    """
  Generate correlated Poisson type spike trains.
  Args:
    data     : 2D array with spike trains of N_Neurons
    num_bins         : number of bins over the total duration

  Returns:
    count_st   : 2D array of spike counts for each train per time bin
  """

    #initiate spike count matrix
    count_st = np.zeros((data.shape[0], int(num_bins)))

    #create bin edge array
    time_bins = np.linspace(0, pars['T'], int(num_bins) + 1)

    #get lv_matrix for referencing
    lv_matrix, _ = modes(pars)

    for i in tqdm(range(data.shape[0]), desc='Binning', unit='step'):
        count_st[i], _ = np.histogram(np.where(data[i] == 1)[0], bins=time_bins)

    #average rows into 4 firing modes
    count_modes = count_st.reshape(lv_matrix.shape[1], data.shape[0] // lv_matrix.shape[1], num_bins)
    count_modes_avg_pre = np.mean(count_modes, axis=1)
    count_modes_avg = count_modes_avg_pre * (1000 / pars['bin_size'])

    #average rows into 160 neurons
    count_trials = count_st.reshape(pars['n_neurons'], pars['trials'], num_bins)
    count_trials_avg_pre = np.mean(count_trials, axis=1)
    count_trials_avg = count_trials_avg_pre * (1000 / pars['bin_size'])

    return count_modes_avg, count_trials_avg


def matrices(count_st):
    #compute var/cov calcs
    CVM = np.cov(count_st, ddof=1)

    #calc pearson coefficients
    corr_matrix = np.corrcoef(count_st)

    return CVM, corr_matrix


def plots(pars,
          range_t,
          spike_trains,
          count_modes_avg,
          x_counts, CVM,
          corr_matrix,
          lv_matrix,
          principal_components,
          count_modes_boot_avg,
          evr_uns,
          evr_uns_cumsum,
          evr_boot,
          evr_boot_cumsum,
          evr_boot_plus,
          evr_boot_minus,
          evr_boot_cumsum_plus,
          evr_boot_cumsum_minus,
          n_it,
          npc,
          int_uns,
          int_boot,
          title,
          p,
          p_cumsum,
          filename='synchrony_simulation_mixed_N120_T100_HighWeights_NoNoise_{}.png'):
    """
    Function generates and plots the raster of the Poisson spike trains,
    overlayed with the line graphs of the spike counts per time bin

    Args:
      pars: parameter dictionary
      range_t     : time sequence
      spike_trains : 2D array with binary spike trains, with shape (N, Lt)
      count_modes_avg: array with averaged counts for each train per time bin
      x_counts: x values of spike counts
      CVM: cov matrix
      corr_matrix: correlation matrix
      lv_matrix: latent variable matrix
      principal_components: principal components
      count_modes_boot_avg: averaged counts for each train per time bin of bootstrapped array
      evr_uns: explained variance ratios of un-shuffled spike counts
      evr_boot: explained variance ratios of bootstrapped spike counts
      evr_boot_plus: upper confidence interval of mean
      evr_boot_minus: lower confidence interval of mean
      n_it: number of iterations
      filename: filename to save the figure
      npc: number of principal components explaining more variance than shuffled
      int_uns: interception of un-shuffled data with npc
      int_boot: interception of shuffled data with npc
      title: title of the figure



    Returns:
      Raster plot of spike trains with line graphs of spike counts per time bin
    """

    # Initiate figure
    fig, axs = plt.subplots(4, 2, figsize=(30, 50))  #30, 40 for N=100
    fig.suptitle(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n'
                 f'{title}', fontsize=40)

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.5)

    #Create colormap for plot coloring
    cmap = plt.cm.get_cmap('viridis', pars['n_neurons'])
    colors = cmap(np.linspace(0, 1, pars['n_neurons']))
    colors_avg_spikes = ['purple', 'blue', 'green', 'yellow', 'black']

    alpha = 1 / pars['trials']

    #smooting indicator

    # Plot primary y-axis (raster plot)
    for neuron in tqdm(range(pars['n_neurons']), desc='Plotting spikes', unit='step'):
        for trial in range(pars['trials']):
            idx = neuron * pars['trials'] + trial
            if spike_trains[idx].sum() > 0.:
                t_st = range_t[spike_trains[idx] > 0.5]  # spike times
                y_position = neuron
                axs[0, 0].plot(t_st, np.ones(len(t_st)) * y_position,
                               '|',
                               color=colors[neuron],
                               ms=7,
                               markeredgewidth=0.5,
                               alpha=alpha)

    # y1 settings
    axs[0, 0].set_xlim([range_t[0], range_t[-1]])
    axs[0, 0].set_ylim([-0.5, pars['n_neurons'] - 0.5])

    yticks = np.arange(0, pars['n_neurons'], 10)
    axs[0, 0].set_yticks(yticks)
    axs[0, 0].set_yticklabels(yticks + 1, fontsize=20)
    axs[0, 0].tick_params(axis='x', labelsize=20)

    axs[0, 0].set_title(f"N = {pars['n_neurons']},"
                        f" Neurons per LV = {pars['n_neurons'] // lv_matrix.shape[1]},"
                        f" Trials per Neuron = {pars['trials']},\n"
                        f" Noise amp = {pars['noise_amp']}",
                        fontsize=30)
    axs[0, 0].set_xlabel('Time (ms)', fontsize=20)
    axs[0, 0].set_ylabel('Neuron ID', fontsize=20)
    #axs[0, 0].legend(loc="upper left")

    # initiate secondary y axis
    y2 = axs[0, 0].twinx()

    # Plot secondary y axis (line graph)
    for i in range(count_modes_avg.shape[0]):
        x_smooth = np.linspace(x_counts.min(), x_counts.max(), 800)
        spl = mis(x_counts, count_modes_avg[i, :], k=5)
        y_smooth = spl(x_smooth)
        y2.plot(x_smooth, y_smooth,
                label=f"count_modes_avg{i + 1}",
                color=colors_avg_spikes[i],
                linewidth=5)

    # y2 settings
    y2.set_xlim([range_t[0], range_t[-1]])
    y2.set_ylabel('Spike counts', fontsize=20)

    y2.tick_params(axis='y', labelsize=20)
    #y2.legend(loc="upper right")

    print('Plotting the rest...')

    #labeling arrays
    ticks = np.arange(0, CVM.shape[0], 10)
    ticklabels = np.arange(1, CVM.shape[0] + 1, 10)

    #Plot CVM
    cax1 = axs[1, 0].matshow(CVM, interpolation='nearest',
                             cmap='coolwarm')

    divider1 = mal(axs[1, 0])
    cbar1_ax = divider1.append_axes('right', size='5%', pad=0.1)
    cbar1 = plt.colorbar(cax1, cax=cbar1_ax)
    cbar1.ax.tick_params(labelsize=20)

    #Settings
    axs[1, 0].set_xticks(ticks)
    axs[1, 0].set_yticks(ticks)
    axs[1, 0].set_xticklabels(ticklabels, fontsize=20)
    axs[1, 0].set_yticklabels(ticklabels, fontsize=20)
    axs[1, 0].set_title('Covariance Matrix', fontsize=30)
    axs[1, 0].set_xlabel('Neuron (all trials)', fontsize=20)
    axs[1, 0].set_ylabel('Neuron (all trials)', fontsize=20)

    #Plot Pearson Matrix
    cax2 = axs[1, 1].matshow(corr_matrix,
                             interpolation='nearest',
                             cmap='coolwarm',
                             vmin=-1, vmax=1)

    divider2 = mal(axs[1, 1])
    cbar2_ax = divider2.append_axes('right', size='5%', pad=0.1)
    cbar2 = plt.colorbar(cax2, cax=cbar2_ax)
    cbar2.ax.tick_params(labelsize=20)

    #Settings
    axs[1, 1].set_xticks(ticks)
    axs[1, 1].set_yticks(ticks)
    axs[1, 1].set_xticklabels(ticklabels, fontsize=20)
    axs[1, 1].set_yticklabels(ticklabels, fontsize=20)
    axs[1, 1].set_title('Pearson Matrix', fontsize=30)
    axs[1, 1].set_xlabel('Neuron (all trials)', fontsize=20)
    axs[1, 1].set_ylabel('Neuron (all trials)', fontsize=20)

    #Plot weights over LV
    weights = np.array([
        [pars['w11'], pars['w12'], pars['w13'], pars['w14']],
        [pars['w21'], pars['w22'], pars['w23'], pars['w24']],
        [pars['w31'], pars['w32'], pars['w33'], pars['w34']],
        [pars['w41'], pars['w42'], pars['w43'], pars['w44']]
    ])

    cax3 = axs[0, 1].matshow(weights.T,
                             interpolation='nearest',
                             cmap='coolwarm',
                             vmin=0, vmax=1)

    divider3 = mal(axs[0, 1])
    cbar3_ax = divider3.append_axes('right', size='5%', pad=0.1)
    cbar3 = plt.colorbar(cax3, cax=cbar3_ax)
    cbar3.ax.tick_params(labelsize=20)

    #Axis settings
    ticks = np.arange(0, lv_matrix.shape[1], 1)
    xlabels = ['Steady State', 'Ramp', 'Sine', 'Shifted Sine']
    ylabels = ['N 1-40', 'N 41-80', 'N 81-120', 'N121-160']

    axs[0, 1].set_xticks(ticks)
    axs[0, 1].set_yticks(ticks)
    axs[0, 1].set_xticklabels(xlabels, fontsize=20)
    axs[0, 1].set_yticklabels(ylabels, fontsize=20)

    axs[0, 1].set_title('Weights', fontsize=30)
    axs[0, 1].set_xlabel('Latent variable', fontsize=20)
    axs[0, 1].set_xlabel('Latent variable', fontsize=20)
    axs[0, 1].set_ylabel('Neuron batch', fontsize=20)

    #Plot PCA results
    x = np.arange(principal_components.shape[1])

    for i in range(5):  #principal_components.shape[0]
        axs[2, 0].plot(x, principal_components[i],
                       label=f"Principal component {i + 1}",
                       linewidth=5,
                       color=colors_avg_spikes[i])
        #               alpha=1/principal_components.shape[0])
        # axs[2, 0].plot(x_bins, principal_components[i],
        #                label=f"Avg PC{i+1}",
        #                linewidth=5,)

    axs[2, 0].set_title(f'PCA, Smoothing factor = {pars['window_size']}', fontsize=30)
    axs[2, 0].set_xlabel('???', fontsize=20)
    axs[2, 0].set_ylabel('Component value', fontsize=20)
    axs[2, 0].legend(loc='upper right')
    axs[2, 0].tick_params(axis='x', labelsize=20)
    axs[2, 0].tick_params(axis='y', labelsize=20)

    # Plot bootstapped data
    for i in range(count_modes_boot_avg.shape[0]):
        axs[2, 1].plot(x_counts, count_modes_boot_avg[i],
                       label=f"{xlabels[i]}",
                       color=colors_avg_spikes[i],
                       linewidth=5)

    # axis settings
    axs[2, 1].set_title(f'Bootstrap', fontsize=30)
    axs[2, 1].legend(loc='upper right')
    axs[2, 1].set_xlim([range_t[0], range_t[-1]])
    axs[2, 1].set_ylabel('Spike counts', fontsize=20)

    axs[2, 1].tick_params(axis='y', labelsize=20)
    axs[2, 1].tick_params(axis='x', labelsize=20)

    #Plot PCA explained variance plot
    icks = np.arange(evr_uns.shape[0])
    y = icks / evr_uns.shape[0]  #this is the straight reference line

    axs[3, 0].plot(icks, evr_uns,
                   label='sorted data',
                   linewidth=5,
                   color='black')
    axs[3, 0].plot(icks, evr_boot,
                   label='bootstrapped data (avg)',
                   linewidth=5,
                   color='gray')
    axs[3, 0].plot(icks, evr_uns_cumsum,
                   label='sorted data cumsum',
                   linewidth=5,
                   color='black',
                   alpha=0.5)
    axs[3, 0].plot(icks, evr_boot_cumsum,
                   label=f'bootstrapped data cumsum (avg)',
                   linewidth=5,
                   color='gray',
                   alpha=0.5)
    axs[3, 0].plot(icks, evr_boot_plus,
                   label=f'upper and lower confidence threshold',
                   linewidth=1,
                   color='blue',
                   markeredgecolor='black',
                   zorder=3,
                   alpha=0.5)
    axs[3, 0].plot(icks, evr_boot_minus,
                   linewidth=1,
                   color='blue',
                   markeredgecolor='black',
                   zorder=3,
                   alpha=0.5)
    axs[3, 0].plot(icks, evr_boot_cumsum_plus,
                   label=f'upper and lower confidence threshold (cumsum)',
                   linewidth=1,
                   color='blue',
                   markeredgecolor='black',
                   zorder=3,
                   alpha=0.5)
    axs[3, 0].plot(icks, evr_boot_cumsum_minus,
                   linewidth=1,
                   color='blue',
                   markeredgecolor='black',
                   zorder=3,
                   alpha=0.5)
    axs[3, 0].plot(icks, y,
                   linewidth=3,
                   linestyle='dashed',
                   alpha=0.5,
                   color='gray')
    axs[3, 0].axvline(x=npc, linestyle='--', color='red', label=f'N(PC) = {npc}')
    axs[3, 0].scatter(npc, int_uns, color='black', edgecolors='red',
                      label=f'{int_uns:.3f}', s=100, zorder=3)
    axs[3, 0].scatter(npc, int_boot, color='gray', edgecolors='red',
                      label=f'{int_boot:.3f}', s=100, zorder=3)

    #axis settings
    axs[3, 0].set_title(f'PCA Explained Variance vs. bootstrap ({n_it} It.)', fontsize=30)
    axs[3, 0].set_xlabel('Principal component', fontsize=20)
    axs[3, 0].set_ylabel('Explained variance (%)', fontsize=20)
    axs[3, 0].legend(loc='upper right')
    axs[3, 0].tick_params(axis='x', labelsize=20)
    axs[3, 0].tick_params(axis='y', labelsize=20)

    #Plot alpha table

    p_cut = np.round(p[:5], 4)
    p_cumsum_cut = np.round(p_cumsum[:5], 4)
    table_data = [list(row) for row in zip(p_cut, p_cumsum_cut)]
    column_labels = ['evr', 'evr_cumsum']
    row_labels = ['PC' + str(i + 1) for i in range(5)]

    # Create the table
    table = None
    for i in range(5):
        table = axs[3, 1].table(cellText=table_data,
                        colLabels=column_labels,
                        rowLabels=row_labels,
                        loc='center',
                        cellLoc='center')

    #axis settings
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    axs[3, 1].set_title('p-values', fontsize=30)
    axs[3, 1].axis('off')

    #Delete empty subplots
    #fig.delaxes(axs[3, 1])

    # #Save & Show final plot
    # index = 1
    # while os.path.exists(filename.format(index)):
    #     index +=1
    #
    # filename = filename.format(index)
    # print(f"Plot saved as {filename}")
    # fig.savefig(filename)

    plt.show()



def main(shuffle_loc):
    ### 0: PREP
    start = time.perf_counter()
    pars = default_pars()
    seed = False
    bin_size = pars['bin_size']  # in ms
    num_bins = pars['T'] // bin_size

    #init evr arrays
    evr_boot_acc = np.zeros((num_bins, pars['n_it']))  # (80, 100)
    evr_boot_cumsum_acc = np.zeros((num_bins, pars['n_it']))  # (80, 100)

    evr_boot_mean = np.zeros((num_bins,))           # (80,)
    evr_boot_stdev = np.zeros((num_bins,))          # (80,)
    evr_boot_cumsum_mean = np.zeros((num_bins,))    # (80,)
    evr_boot_cumsum_stdev = np.zeros((num_bins,))   # (80,)
    evr_p = np.zeros((num_bins,))               # (80,)
    evr_cumsum_p = np.zeros((num_bins,))        # (80,)

    ### 1: SPIKE GENERATION
    lv_matrix, wgt_matrix = modes(pars)
    acc_spike_trains = multiple_poisson_generator(pars, myseed=seed)

    if shuffle_loc == 1:
        title = 'bin --> smooth --> shuffle'
        ### 2: BINNING
        count_modes_avg, count_trials_avg = bin_spike_times(pars, num_bins, data=acc_spike_trains)

        ### 3: SMOOTHING
        count_trials_avg_smt = smoothing(pars, data=count_trials_avg)

        ### 4: SHUFFLING & PCA

        #init shuffled mode count array
        count_modes_avg_shf_acc = np.zeros(count_modes_avg.shape)

        for i in tqdm(range(pars['n_it']), desc="Analyzing", unit="step"):
            #trials
            count_trials_avg_smt_shf = bootstrap(data=count_trials_avg_smt)

            #modes
            count_modes_avg_smt_shf = bootstrap(data=count_modes_avg)
            count_modes_avg_shf_acc += count_modes_avg_smt_shf

            #pca (evr_boot)
            _, evr_boot, evr_boot_cumsum = pca(data=count_trials_avg_smt_shf)
            evr_boot_acc[:, i] = evr_boot
            evr_boot_cumsum_acc[:, i] = evr_boot_cumsum

        #average modes array
        count_modes_avg_shf = count_modes_avg_shf_acc / pars['n_it']

        ### 5:  PCA (un-shuffled)
        principal_components, evr_uns, evr_uns_cumsum = pca(data=count_trials_avg_smt)

        #generate statistics arrays
        evr_boot_mean, evr_boot_stdev, evr_p = statistics(evr_boot_acc, evr_uns)
        evr_boot_cumsum_mean, evr_boot_cumsum_stdev, evr_cumsum_p = statistics(evr_boot_cumsum_acc, evr_uns_cumsum)


    elif shuffle_loc == 2:
        title = 'bin --> shuffle --> smooth'
        ### 2: BINNING
        count_modes_avg, count_trials_avg = bin_spike_times(pars, num_bins, data=acc_spike_trains)

        ### 3: SHUFFLING, SMOOTHING & PCA
        # init shuffled mode count array
        count_modes_avg_shf_acc = np.zeros(count_modes_avg.shape)

        for i in tqdm(range(pars['n_it']), desc="Analyzing", unit="step"):
            # trials
            count_trials_avg_shf = bootstrap(data=count_trials_avg)

            #smoothen
            count_trials_avg_shf_smt = smoothing(pars, data=count_trials_avg_shf)

            # modes
            count_modes_avg_shf = bootstrap(data=count_modes_avg)
            count_modes_avg_shf_acc += count_modes_avg_shf

            # pca (evr_boot)
            _, evr_boot, evr_boot_cumsum = pca(data=count_trials_avg_shf_smt)
            evr_boot_acc[:, i] = evr_boot
            evr_boot_cumsum_acc[:, i] = evr_boot_cumsum

        # average modes array
        count_modes_avg_shf = count_modes_avg_shf_acc / pars['n_it']

        #smoothen un-shuffled data
        count_trials_avg_smt = smoothing(pars, data=count_trials_avg)

        ### 5:  PCA (un-shuffled)
        principal_components, evr_uns, evr_uns_cumsum = pca(data=count_trials_avg_smt)

        # generate statistics arrays
        evr_boot_mean, evr_boot_stdev, evr_p = statistics(evr_boot_acc, evr_uns)
        evr_boot_cumsum_mean, evr_boot_cumsum_stdev, evr_cumsum_p = statistics(evr_boot_cumsum_acc, evr_uns_cumsum)


    elif shuffle_loc == 3:
        title = 'shuffle --> bin --> smooth'

        #un-shuffled data

        #binning
        count_modes_avg, count_trials_avg = bin_spike_times(pars, num_bins,
                                                            data=acc_spike_trains)

        #smoothing
        count_trials_avg_smt = smoothing(pars, data=count_trials_avg)

        #pca
        principal_components, evr_uns, evr_uns_cumsum = pca(data=count_trials_avg_smt)

        #init shuffle arrays
        count_modes_avg_shf = np.zeros((count_modes_avg.shape))

        ### 2: SHUFFLING, BINNING, SMOOTHING, PCA
        for i in tqdm(range(pars['n_it']), desc="Analyzing", unit="step"):
            acc_spike_trains_shf = bootstrap(data=acc_spike_trains)

            #binning
            count_modes_avg_shf, count_trials_avg_shf = bin_spike_times(pars, num_bins,
                                                                                data=acc_spike_trains_shf)

            #smoothing
            count_trials_avg_shf_smt = smoothing(pars, data=count_trials_avg_shf)

            #pca
            _, evr_boot, evr_boot_cumsum = pca(data=count_trials_avg_shf_smt)
            evr_boot_acc[:, i] = evr_boot
            evr_boot_cumsum_acc[:, i] = evr_boot_cumsum

            # generate statistics arrays
            evr_boot_mean, evr_boot_stdev, evr_p = statistics(evr_boot_acc, evr_uns)
            evr_boot_cumsum_mean, evr_boot_cumsum_stdev, evr_cumsum_p = statistics(evr_boot_cumsum_acc, evr_uns_cumsum)


    else:
        raise ValueError("Invalid shuffle_loc value. Must be 1, 2, or 3.")

    # Calculate number of PCs explaining more variance than shuffled
    npc = 0
    for i in range(len(evr_uns)):
        if evr_uns[i] > evr_boot_mean[i]:
            npc += 1
        else:
            break
    int_uns = evr_uns_cumsum[npc]
    int_boot = evr_boot_cumsum_mean[npc]

    #Calculate confidence thresholds
    evr_boot_plus = evr_boot_mean + evr_boot_stdev
    evr_boot_minus = evr_boot_mean - evr_boot_stdev

    evr_boot_cumsum_plus = evr_boot_cumsum_mean + evr_boot_cumsum_stdev
    evr_boot_cumsum_minus = evr_boot_cumsum_mean - evr_boot_cumsum_stdev

    ### PLOTTING
    x_counts = np.linspace(bin_size / 2, pars['T'] - bin_size / 2, num_bins)
    CVM, corr_matrix = matrices(count_st=count_trials_avg)

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
