import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import sys
import scipy
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D

# Path and num runs value for evaluation
current = "2017-11-09_17-22-00_hamming"
num_runs = 10  # How many runs should we evaluate

# Prepare path and get files
path = os.getcwd() + "/backup/test_multi/" + current
datapath = path + "/data"
plotpath = path + "/plots"

if not os.path.exists(plotpath):
    os.mkdir(plotpath)

# Import parameters
sys.path.insert(0,path+"/michaelis")
sys.path.insert(0,os.getcwd()+"/utils")
import param_mcmc_multi as para
import stationary as stationaryDistribution

sources = { 'transition_distances': None, 'activity': None, 'estimated_stationaries': None, 'ncomparison': None, 'hamming_distances': None }

# Prepare data from files to numpy array
def prepare_data(sources):
    data = sources
    for d in sources:
        # Get folder and store files in arrays
        folder = glob.glob(os.path.join(datapath, d))[0]
        arrays = [np.load(folder + '/run' + str(run) + '.npy') for run in range(num_runs)]
        # Stack arrays to one array
        data[d] = np.stack(arrays, axis=0)
    return data

def training_steps_plot(distances, suffix, ytext):
    global plotpath

    # runs, models, train steps, test chunks, dists
    # Get mean over "runs" and over "test chunks"
    # CAUTION: Mean over "test steps" is only appropriate if STDP is switched off in test phase
    dists_mean = np.mean(distances, axis=(0,3))
    dists_std = np.std(distances, axis=(0,3))

    # Mean just over test chunks, not over runs
    dists_mean_single = np.mean(distances, axis=3)

    # Get number of models and calculate train steps
    num_models = np.shape(distances)[1]

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, num_models))

    # Plot influence of training steps for every model
    for i in range(num_models):
        legend = 'Model '+str(i+1)
        plt.errorbar(para.c.steps_plastic, dists_mean[i,:], label=legend, yerr=dists_std[i], color=color_palette[i],
                     elinewidth=1, ecolor=np.append(color_palette[i][0:3], 0.5))

        # Plot all runs of every model (transparent in background)
        for j in range(np.shape(dists_mean_single)[0]):
            plt.plot(para.c.steps_plastic, dists_mean_single[j, i], color=color_palette[i], alpha=0.1)

    # Beautify plot and save png file
    plt.legend()
    plt.ylim(ymin=0)
    plt.xlabel('Training steps')
    plt.ylabel(ytext)
    plt.savefig(plotpath + '/distances_training_steps_'+suffix+'.png', dpi=144)
    plt.close()

def training_steps_plot_thresholds(distances):
    global plotpath

    # runs, models, train steps, thresholds, test steps
    # Get mean over "runs" and over "test steps"
    # CAUTION: Mean over "test steps" is only appropriate if STDP is switched off in test phase
    dists_mean = np.mean(distances, axis=(0, 4))
    # Format after: models, train steps, thresholds

    # Get number of models and number of thresholds
    num_models = np.shape(distances)[1]
    num_thresh = np.shape(distances)[3]

    # Define color palette
    thresh_colors = cm.rainbow(np.linspace(0, 1, num_thresh))
    model_colors = cm.rainbow(np.linspace(0, 1, num_models))

    # Plot influence of training steps for every model
    for i in range(num_models):
        for j in range(num_thresh):
            opacity = 1 if j == num_thresh-1 else 0.2
            legend = None
            if j == 0:
                legend = 'Model '+str(i+1)+', Threshold finite'
            elif j == num_thresh-1:
                legend = 'Model '+str(i+1)+', Threshold inf'
            plt.errorbar(para.c.steps_plastic, dists_mean[i,:,j], label=legend, color=np.append(model_colors[i][0:3], opacity))
                         #path_effects=[pe.SimpleLineShadow(shadow_color=thresh_colors[j]), pe.Normal()])

    # Beautify plot and save png file
    plt.legend(prop={'size': 6})
    plt.ylim(ymin=0)
    plt.xlabel('Training steps')
    plt.ylabel('Error')
    plt.savefig(plotpath + '/distances_training_steps_with_thresholds.png', dpi=144)
    plt.close()

def test_trace_plot(distances, suffix, label):
    global plotpath, para

    # Get results for highest train step only
    last_idx_train_steps = np.shape(distances)[2]-1
    dists = distances[:,:,last_idx_train_steps,:]

    # Calculate mean and standard deviation
    dists_mean = np.mean(dists, axis=0)
    dists_std = np.std(dists, axis=0)

    # Get number of original test steps (for x axis)
    test_steps = np.arange(np.shape(dists)[2]) * para.c.stats.transition_step_size

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, np.shape(dists_mean)[0]))

    # Plot mean of every model
    for i in range(np.shape(dists)[1]):
        legend = 'Model ' + str(i + 1)
        plt.errorbar(test_steps, dists_mean[i], label=legend, yerr=dists_std[i], color=color_palette[i],
                     elinewidth=1, ecolor=np.append(color_palette[i][0:3], 0.5))

        # Plot all runs of every model (transparent in background)
        for j in range(np.shape(dists)[0]):
            plt.plot(test_steps, dists[j,i], color=color_palette[i], alpha=0.1)

    # Beautify plot and save png file
    plt.legend()
    plt.ylim(ymin=0)
    plt.xlabel('Test steps')
    plt.ylabel(label)
    plt.savefig(plotpath + '/test_traces_'+suffix+'.png', dpi=144)
    plt.close()

def activity_distance_correlation_plot(distances, activity):
    global plotpath, test_step_size

    # Get results for highest train step only
    last_idx_train_steps = np.shape(distances)[2] - 1
    dists = distances[:, :, last_idx_train_steps, :]
    actis = activity[:, :, last_idx_train_steps, :]

    # x = np.mean(dists[:,:,:], axis=(0,2)).flatten()
    # y = np.mean(actis[:,:,:], axis=(0,2)).flatten()
    # plt.scatter(x, y)

    for i in range(np.shape(dists)[1]):
        plt.figure()
        x = dists[:,i,:].flatten()
        y = actis[:,i,:].flatten()
        plt.scatter(x, y)
        plt.title('Correlation Activity/Distances (%.2f)' % pearsonr(x, y)[0])
        plt.xlabel('Mean squared distance to initial transition')
        plt.ylabel('Activity (percentage)')
        plt.savefig(plotpath + '/correlation_activity_model' + str(i+1) + '.png', dpi=144)
        plt.close()

def ncomparison_distance_correlation_plot(distances, ncomparison):
    global plotpath, test_step_size

    # Get results for highest train step only
    last_idx_train_steps = np.shape(distances)[2] - 1
    dists = distances[:, :, last_idx_train_steps, :]
    ncomp = ncomparison[:, :, last_idx_train_steps]

    num_models = np.shape(dists)[1]

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, num_models))

    # Plot
    plt.figure()
    for i in range(num_models):
        x = np.mean(dists[:, i, 1:], axis=1) # Exclude first test step
        y = ncomp[:, i]
        plt.scatter(x, y, color=color_palette[i], alpha=0.3)
        plt.errorbar(np.mean(x), np.mean(y), yerr=np.std(y), fmt='o', color=color_palette[i])

    # Add decoration
    plt.title('Correlation NComparison/Distances (%.2f)' % pearsonr(np.mean(ncomp, axis=0), np.mean(dists, axis=(0,2)))[0])
    plt.xlabel('Mean squared distance to initial transition')
    plt.ylabel('Number of comparison states')

    # Save and close plit
    plt.savefig(plotpath + '/correlation_ncomparison_distances.png', dpi=144)
    plt.close()

def hamming_histogram(hamming_distances):
    global plotpath

    # Mean over runs
    hamming_mean = np.mean(hamming_distances, axis=0)
    # Form after: models, train steps, hammings

    # Get largest, middle and smallest training run
    hms = np.shape(hamming_mean)
    hmean_maxmin = np.empty((hms[0], 3, hms[2]))
    maxmin_idx = np.array([0, (hms[1]-1)/2, hms[1]-1])
    hmean_maxmin[:, 0, :] = hamming_mean[:, maxmin_idx[0], :]
    hmean_maxmin[:, 1, :] = hamming_mean[:, maxmin_idx[1], :]
    hmean_maxmin[:, 2, :] = hamming_mean[:, maxmin_idx[2], :]
    # Form after: models, min/middle/max train steps, hammings

    # Plot preparation
    bar_width = 0.15  # width of bar
    intra_bar_space = 0.05  # space between multi bars
    coeff = np.array([-1, 0, 1])  # left, middle, right from tick

    # Get overal max hamming value and frequency, used for axis
    max_freq = np.max([np.max(np.bincount(hmean_maxmin[i, j, :].astype(int))) for j in range(3) for i in range(hms[0])])
    max_val = np.max(hmean_maxmin)

    # Check normality (TODO already calculated, can be used if necessary)
    #pvalues = [scipy.stats.kstest(hmean_maxmin[i, j, :], 'norm')[0] for j in range(3) for i in range(hms[0])]

    for i in range(hms[0]):
        for j in range(3):
            # Calculate frequencies
            hamming_freqs = np.bincount(hmean_maxmin[i, j, :].astype(int))
            # Calculate x positions
            x = np.arange(len(hamming_freqs)) + (coeff[j] * (bar_width + intra_bar_space))
            # Plot barplot with legend
            legend = str(para.c.steps_plastic[maxmin_idx[j]]) + ' training steps'
            plt.bar(x, hamming_freqs, bar_width, label=legend)
        # Adjust plot
        plt.ylim(ymax=max_freq)
        plt.xlim(xmax=max_val)
        plt.legend()
        plt.title('Model: ' + str(i+1))
        plt.xlabel('Hamming distance')
        plt.ylabel('Frequency')
        plt.savefig(plotpath + '/hamming_freqs_model'+str(i+1)+'.png', dpi=144)
        plt.close()

def inequality_distance_correlation_plot(distances):
    global plotpath, para

    # Calculate stationary distributions from markov chains
    stationaries = np.array([stationaryDistribution.calculate(transition) for transition in para.c.source.transitions])

    # Get variance, entropy and gini
    states = np.arange(np.shape(stationaries)[1])+1
    variance = np.sum(np.multiply(stationaries, (states - np.mean(states)) ** 2), axis=1)
    entropy = [scipy.stats.entropy(s, np.repeat(0.25, np.shape(para.c.source.transitions)[1])) for s in stationaries]
    ginis = [calc_gini(x) for x in stationaries]

    # Get number of train steps
    train_steps = np.shape(distances)[2]

    # Exclude first test step and mean over test steps
    dists = np.mean(distances[:, :, :, 1:], axis=3)

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, train_steps))

    # Variance
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, np.mean(dists[:,:,i], axis=0))[0],2))
        plt.errorbar(variance, np.mean(dists[:,:,i], axis=0), label=legend, yerr=np.std(dists[:,:,i], axis=0),  # fmt='o',
                     color=color_palette[i], ecolor=np.append(color_palette[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': 6})
    plt.ylim(ymin=0)
    plt.grid()
    plt.title('Variance/Distances')
    plt.xlabel('Variance')
    plt.ylabel('Mean squared distance to initial transition')
    plt.savefig(plotpath + '/correlation_inequality_variance.png', dpi=144)
    plt.close()

    # Variance baseline plot
    if para.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
        dists_baseline = np.mean(dists[:,:,0], axis=0)
        for i in range(train_steps):
            if i > 0:
                # Variance with distance difference
                diff = dists_baseline - np.mean(dists[:, :, i], axis=0)
                legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, diff)[0], 2))
                plt.plot(variance, diff, label=legend, color=color_palette[i])

        plt.legend(prop={'size': 6})
        #plt.ylim(ymin=0)
        plt.grid()
        plt.title('Baseline: Variance/Distances')
        plt.xlabel('Variance')
        plt.ylabel('Performance increase in relation to baseline')
        plt.savefig(plotpath + '/correlation_inequality_variance_baseline.png', dpi=144)
        plt.close()

    # Entropy
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(entropy, np.mean(dists[:,:,i], axis=0))[0],2))
        plt.errorbar(entropy, np.mean(dists[:,:,i], axis=0), label=legend, yerr=np.std(dists[:,:,i], axis=0),  # fmt='o',
                     color=color_palette[i], ecolor=np.append(color_palette[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': 6})
    plt.ylim(ymin=0)
    plt.grid()
    plt.title('KL/Distances')
    plt.xlabel('KL')
    plt.ylabel('Mean squared distance to initial transition')
    plt.savefig(plotpath + '/correlation_inequality_kl.png', dpi=144)
    plt.close()

    # Entropy baseline plot
    if para.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
        dists_baseline = np.mean(dists[:, :, 0], axis=0)
        for i in range(train_steps):
            if i > 0:
                # Variance with distance difference
                diff = dists_baseline - np.mean(dists[:, :, i], axis=0)
                legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(entropy, diff)[0], 2))
                plt.plot(entropy, diff, label=legend, color=color_palette[i])

        plt.legend(prop={'size': 6})
        # plt.ylim(ymin=0)
        plt.grid()
        plt.title('Baseline: KL/Distances')
        plt.xlabel('KL')
        plt.ylabel('Performance increase in relation to baseline')
        plt.savefig(plotpath + '/correlation_inequality_kl_baseline.png', dpi=144)
        plt.close()

    # Gini
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(ginis, np.mean(dists[:,:,i], axis=0))[0],2))
        plt.errorbar(ginis, np.mean(dists[:,:,i], axis=0), label=legend, yerr=np.std(dists[:,:,i], axis=0),  # fmt='o',
                     color=color_palette[i], ecolor=np.append(color_palette[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': 6})
    plt.ylim(ymin=0)
    plt.grid()
    plt.title('Gini/Distances')
    plt.xlabel('Gini')
    plt.ylabel('Mean squared distance to initial transition')
    plt.savefig(plotpath + '/correlation_inequality_gini.png', dpi=144)
    plt.close()

def calc_gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def lorenz_plot(normed_stationaries):
    global plotpath, para
    # runs, models, train steps, test chunks, stationary

    # Calculate stationary distributions from markov chains
    stationaries = np.array([stationaryDistribution.calculate(transition) for transition in para.c.source.transitions])

    # Preparation
    n = np.shape(stationaries)[1]
    num_models = np.shape(stationaries)[0]
    x = np.repeat(1 / float(n), n)

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, num_models))

    x0 = np.append(0, np.cumsum(x))
    plt.plot(x0, x0, color='black')

    for i in range(num_models):
        y = np.sort(stationaries[i])
        y0 = np.append(0, np.cumsum(y))
        plt.plot(x0, y0, label= 'Model'+str(i+1), color=color_palette[i])

    plt.legend(loc=2, prop={'size': 6})
    plt.ylim(ymax=1)
    plt.grid()
    plt.title('Lorenz curve: Stationary distributions')
    plt.xlabel('States')
    plt.ylabel('Cumulative probability')
    plt.savefig(plotpath + '/lorenz-curve.png', dpi=144)
    plt.close()

def get_max_threshold(arr):
    return arr[:,:,:,np.shape(arr)[3]-1,:]

def norm_stationaries(estimated_stationaries):
    # Get values for max threshold
    estimated_stationaries = get_max_threshold(estimated_stationaries)

    # Normalize
    est_norm = estimated_stationaries / np.sum(estimated_stationaries, axis=4)[:, :, :, :, np.newaxis]

    return est_norm

def stationariy_distances(est_norm):
    global para

    # Calculate stationaries
    T = para.c.source.transitions
    stationaries = np.stack([stationaryDistribution.calculate(trans) for trans in T])

    # Calculate distances
    # runs, models, train steps, thresholds, test chunks, stationary
    diff = est_norm - stationaries[np.newaxis, :, np.newaxis, np.newaxis, :]
    distances = np.sum(np.square(diff), axis=4)

    return distances

def prepare_hamming(hamming_distances):
    global para

    # Get values for max threshold
    hamming_distances = get_max_threshold(hamming_distances)
    # Form after: runs, models, train steps, hammings

    # Separate in test chunks
    chunks = para.c.steps_noplastic_test/para.c.stats.transition_step_size
    hamming_distances = np.moveaxis(np.split(hamming_distances, chunks, axis= 3), 0, 3)
    # Form after: runs, models, train steps, test chunks, hammings

    # Calculate hamming means
    return np.mean(hamming_distances, axis=4) / para.c.N_e

##################################################
#################### Evaluate ####################
##################################################

# transition_distances, activity, estimated_stationaries, ncomparison, hamming_distances

#################### Prepare data ####################

data = prepare_data(sources)  # runs, models, train steps, thresholds, test steps / test chunks

normed_stationaries = norm_stationaries(data['estimated_stationaries'])
stationairy_distances = stationariy_distances(normed_stationaries)
hamming_means = prepare_hamming(data['hamming_distances'])
#ncomparison = get_max_threshold(data['ncomparison'])

#################### Training step plots ####################

# Plot transition performance for different training steps
if np.shape(data['transition_distances'])[2] > 1:
    training_steps_plot(get_max_threshold(data['transition_distances']),
        suffix="transition", ytext="Mean squared distance to initial transition")

# Plot stationary performance for different training steps
if np.shape(data['estimated_stationaries'])[2] > 1:
    training_steps_plot(stationairy_distances,
        suffix = "stationary", ytext = "Mean squared distance to stationary")

# Plot hamming mean for different training steps
if np.shape(data['hamming_distances'])[2] > 1:
    training_steps_plot(hamming_means,
        suffix="hamming", ytext="Relative mean hamming distance")

#################### Test chunk plots ####################

test_trace_plot(get_max_threshold(data['transition_distances']),
                suffix="distances", label="Mean squared distance to initial transition")
test_trace_plot(stationairy_distances,
                suffix="stationary", label="Mean squared distance to stationary")
#test_trace_plot(get_max_threshold(activity),
#                suffix="activity", label="Activity (percentage)")

#################### Hamming Distancs evaluation ####################

hamming_histogram(get_max_threshold(data['hamming_distances']))
training_steps_plot_thresholds(data['transition_distances'])

#################### Activity/NComparison ####################

#activity_distance_correlation_plot(distances, activity)
#ncomparison_distance_correlation_plot(get_max_threshold(data['transition_distances']), ncomparison[:,:,:,np.shape(ncomparison)[3]-1])

#################### Variance, Entropy, Gini / Lorenz####################

inequality_distance_correlation_plot(get_max_threshold(data['transition_distances']))
lorenz_plot(normed_stationaries)
