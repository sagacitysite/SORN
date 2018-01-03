import numpy as np
import os
import glob
import sys
import types
import scipy
from scipy.stats import pearsonr

# Import and initalize matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D

#rcParams.keys()

fig_color = '#000000'
legend_size = 9

rcParams['font.family'] = 'CMU Serif'
rcParams['font.size'] = '12'
rcParams['text.color'] = fig_color
rcParams['axes.edgecolor'] = fig_color
rcParams['xtick.color'] = fig_color
rcParams['ytick.color'] = fig_color
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = ':'
rcParams['grid.linewidth'] = 0.5
rcParams['grid.color'] = fig_color
rcParams['legend.fancybox'] = True
rcParams['legend.framealpha'] = 0

# Path and num runs value for evaluation
ip = sys.argv[2] if len(sys.argv) == 3 else 'h_ip_range' # h_ip_factor, eta_ip, h_ip_range
num_runs = 20  # How many runs should we evaluate

# Prepare path and get files
os.chdir("../backup/test_multi/" + str(sys.argv[1]))  #"2017-12-31_12-01-55_models5"
path = os.getcwd()
sys.path.append(path)

datapath = path + "/data"
plotpath = path + "/plots"
if not os.path.exists(plotpath):
    os.mkdir(plotpath)

# Import utils
import utils
def backup_overwrite(a, b=None):
    pass
utils.backup = backup_overwrite  # Overwrite utils backup function to avoid new backup

# Import parameters
#import param_mcmc_multi as para
import imp
para = imp.load_source('param_mcmc_multi', path+'/michaelis/param_mcmc_multi.py')
import utils.stationary as stationaryDistribution

sources = { 'transition_distances': None, 'activity': None, 'estimated_stationaries': None, 'ncomparison': None, 'hamming_distances': None, 'weights_ee': None, 'weights_eu': None }

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
    plt.legend(prop={'size': legend_size})
    plt.ylim(ymin=0)
    plt.xlabel('Training stpdf', color=fig_color)
    plt.ylabel(ytext, color=fig_color)
    plt.savefig(plotpath + '/distances_training_steps_'+suffix+'.pdf', format='pdf', transparent=True)
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
    plt.legend(prop={'size': legend_size})
    plt.ylim(ymin=0)
    plt.xlabel('Training stpdf', color=fig_color)
    plt.ylabel('Error', color=fig_color)
    plt.savefig(plotpath + '/distances_training_steps_with_thresholds.pdf', format='pdf', transparent=True)
    plt.close()

def test_trace_plot(distances, suffix, ylabel, title=None, ymax=None):
    global plotpath, para
    # runs, models, train steps, test steps / test chunks

    train_max = distances[:, :, np.shape(distances)[2]-1]
    # runs, models, test chunks

    # If there is more than one training step, also evaluate train_min as baseline
    train_min = None
    if np.shape(distances)[2] > 1:
        train_min = distances[:, :, 0]
        # runs, models, test chunks

    # Calculate mean and standard deviation
    dists_mean = np.mean(train_max, axis=0)
    dists_std = np.std(train_max, axis=0)

    # Get number of original test steps (for x axis)
    test_steps = np.arange(np.shape(train_max)[2]) * para.c.stats.transition_step_size + para.c.stats.transition_step_size
    num_models = np.shape(train_max)[1]

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, np.shape(dists_mean)[0]))

    # Plot mean of every model
    for i in range(num_models):
        legend = 'Model ' + str(i + 1)
        plt.errorbar(test_steps, dists_mean[i,:], label=legend, yerr=dists_std[i], color=color_palette[i],
                     elinewidth=1, ecolor=np.append(color_palette[i][0:3], 0.5))

        # Plot all runs of every model (transparent in background)
        for j in range(np.shape(train_max)[0]):
            plt.plot(test_steps, train_max[j,i], color=color_palette[i], alpha=0.1)

    # Beautify plot and save png file
    plt.legend(prop={'size': legend_size})
    if title:
        plt.title(title)
    if ymax:
        plt.ylim(ymax=ymax)
    plt.ylim(ymin=0)
    plt.xlabel('Test stpdf', color=fig_color)
    plt.ylabel(ylabel, color=fig_color)
    plt.savefig(plotpath + '/test_traces_'+suffix+'.pdf', format='pdf', transparent=True)
    plt.close()

    # Barplot
    shift = 0.75 if train_min is not None else 0.5
    bar_width = 0.25 if train_min is not None else 0.5
    x = np.arange(num_models) + shift
    da = train_max[:, :, np.shape(train_max)[2]-1]
    plt.bar(x+0.25, np.mean(da, axis=0),
            bar_width, linewidth=0, yerr=np.std(da, axis=0))
    if train_min is not None:
        da = train_min[:, :, np.shape(train_min)[2]-1]
        plt.bar(x, np.mean(da, axis=0),
                bar_width, color='red', linewidth=0,
                yerr=np.std(da, axis=0))
    if title:
        plt.title(title)
    if ymax:
        plt.ylim(ymax=ymax)
    plt.ylim(ymin=0)
    plt.xlabel('Model', color=fig_color)
    plt.ylabel(ylabel, color=fig_color)
    plt.savefig(plotpath + '/performance_' + suffix + '.pdf', format='pdf', transparent=True)
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
        plt.xlabel('Mean squared distance to initial transition', color=fig_color)
        plt.ylabel('Activity (percentage)', color=fig_color)
        plt.savefig(plotpath + '/correlation_activity_model' + str(i+1) + '.pdf', format='pdf', transparent=True)
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
    plt.xlabel('Mean squared distance to initial transition', color=fig_color)
    plt.ylabel('Number of comparison states', color=fig_color)

    # Save and close plit
    plt.savefig(plotpath + '/correlation_ncomparison_distances.pdf', format='pdf', transparent=True)
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

    # Get overal max hamming value and frequency, used for axis
    max_freq = np.max([np.max(np.bincount(hmean_maxmin[i, j, :].astype(int))) for j in range(3) for i in range(hms[0])])
    max_val = np.max(hmean_maxmin)

    # Barplot preparation
    bar_width = 0.15  # width of bar
    intra_bar_space = 0.05  # space between multi bars
    coeff = np.array([-1, 0, 1])  # left, middle, right from tick

    # Define color palette
    color_palette = cm.copper(np.linspace(0, 1, 3))

    # Check normality (TODO already calculated, can be used if necessary)
    #pvalues = [scipy.stats.kstest(hmean_maxmin[i, j, :], 'norm')[0] for j in range(3) for i in range(hms[0])]

    for i in range(hms[0]):
        for j in range(3):
            # Calculate frequencies
            hamming_freqs = np.bincount(hmean_maxmin[i, j, :].astype(int))
            # Calculate x positions
            x = np.arange(len(hamming_freqs)) + (coeff[j] * (bar_width + intra_bar_space))
            # Plot barplot with legend
            legend = str(para.c.steps_plastic[maxmin_idx[j]]) + ' training stpdf'
            plt.bar(x, hamming_freqs, bar_width, label=legend, linewidth=0, color=color_palette[j])
        # Adjust plot
        plt.ylim(ymax=max_freq)
        plt.xlim(xmin=0, xmax=max_val)
        plt.legend(prop={'size': legend_size})
        plt.title('Model: ' + str(i+1))
        plt.xlabel('Hamming distance', color=fig_color)
        plt.ylabel('Frequency', color=fig_color)
        plt.savefig(plotpath + '/hamming_freqs_model'+str(i+1)+'.pdf', format='pdf', transparent=True)
        plt.close()

def inequality_distance_correlation_plot(distances, hpos_idx):
    global plotpath, para
    # runs, models, train steps, h_ip, test chunks

    # Calculate stationary distributions from markov chains
    stationaries = np.array([stationaryDistribution.calculate(transition) for transition in para.c.source.transitions])

    # Get variance, entropy and gini
    states = np.arange(np.shape(stationaries)[1])+1
    variance = np.var(stationaries, axis=1)
    variance[variance < 1e-16] = 0
    #variance = np.sum(np.multiply(stationaries, (states - np.mean(states)) ** 2), axis=1)
    entropy = np.array([scipy.stats.entropy(s, np.repeat(0.25, np.shape(para.c.source.transitions)[1])) for s in stationaries])
    entropy[entropy < 1e-16] = 0
    ginis = np.array([calc_gini(x) for x in stationaries])
    ginis[ginis < 1e-15] = 0
    traces = np.array([np.trace(t) for t in para.c.source.transitions])

    # Get number of train steps
    num_models = np.shape(distances)[1]
    train_steps = np.shape(distances)[2]
    hip_steps = np.shape(distances)[3]

    # Exclude first test step and mean over test steps
    dists = np.mean(distances[:, :, :, :, 1:], axis=4)

    # Define color palette
    train_colors = cm.rainbow(np.linspace(0, 1, train_steps))
    hip_colors = cm.rainbow(np.linspace(0, 1, hip_steps))

    # Trace (highest train, standard h_ip)
    plt.errorbar(traces, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0), fmt='o')

    plt.ylim(ymin=0)
    #plt.grid()
    plt.title('Traces/Distances')
    plt.xlabel('Trace', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_trace.pdf', format='pdf', transparent=True)
    plt.close()

    # Variance (highest train, standard h_ip)
    plt.errorbar(variance, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0) , fmt='o')

    plt.ylim(ymin=0)
    #plt.grid()
    plt.title('Variance/Distances')
    plt.xlabel('Variance', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_variance.pdf', format='pdf', transparent=True)
    plt.close()

    # Variance train
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(variance, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0), fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': legend_size})
    plt.ylim(ymin=0)
    #plt.grid()
    plt.title('Variance/Distances')
    plt.xlabel('Variance', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_variance_train.pdf', format='pdf', transparent=True)
    plt.close()

    # Variance h_ip
    hips = None
    if ip == 'h_ip':
        hips = np.round(np.mean(para.c[ip], axis=0), 3)
    else:
        hips = para.c[ip]

    if len(hips) > 1:
        for i in range(hip_steps):
            errors = np.mean(dists[:,:,train_steps-1,i], axis=0)
            legend = str(hips[i]) + ' ' + str(ip) + ', abs=' + str(np.round(np.abs(np.max(errors) - np.min(errors)), 3))
            plt.errorbar(variance, np.mean(dists[:,:,train_steps-1,i], axis=0), label=legend, yerr=np.std(dists[:,:,train_steps-1,i], axis=0),  fmt='o',
                         color=hip_colors[i], ecolor=np.append(hip_colors[i][0:3], 0.5))

        plt.legend(loc=2, prop={'size': legend_size})
        plt.ylim(ymin=0)
        #plt.grid()
        plt.title('Variance/Distances')
        plt.xlabel('Variance', color=fig_color)
        plt.ylabel('Transition error', color=fig_color)
        plt.savefig(plotpath + '/correlation_inequality_variance_hip.pdf', format='pdf', transparent=True)
        plt.close()

    # Variance baseline plot
    if para.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
        dists_baseline = np.mean(dists[:,:,0], axis=0)
        for i in range(train_steps):
            if i > 0:
                # Variance with distance difference
                diff = dists_baseline.flatten() - np.mean(dists[:, :, i,hpos_idx], axis=0)
                legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, diff)[0], 2))
                plt.plot(variance, diff, label=legend, color=train_colors[i])

        plt.legend(prop={'size': legend_size})
        #plt.ylim(ymin=0)
        #plt.grid()
        plt.title('Baseline: Variance/Distances')
        plt.xlabel('Variance', color=fig_color)
        plt.ylabel('Performance increase in relation to baseline', color=fig_color)
        plt.savefig(plotpath + '/correlation_inequality_variance_train_baseline.pdf', format='pdf', transparent=True)
        plt.close()

    # Entropy (highest train, standard h_ip)
    plt.errorbar(entropy, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0), fmt='o')

    plt.ylim(ymin=0)
    #plt.grid()
    plt.title('KL/Distances')
    plt.xlabel('KL', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_kl.pdf', format='pdf', transparent=True)
    plt.close()

    # Entropy train
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(entropy, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(entropy, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': legend_size})
    plt.ylim(ymin=0)
    #plt.grid()
    plt.title('KL/Distances')
    plt.xlabel('KL', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_kl_train.pdf', format='pdf', transparent=True)
    plt.close()

    # Entropy baseline plot
    if para.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
        dists_baseline = np.mean(dists[:, :, 0, hpos_idx], axis=0)
        for i in range(train_steps):
            if i > 0:
                # Variance with distance difference
                diff = dists_baseline - np.mean(dists[:, :, i,hpos_idx], axis=0)
                legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(entropy, diff)[0], 2))
                plt.plot(entropy, diff, label=legend, color=train_colors[i])

        plt.legend(prop={'size': legend_size})
        # plt.ylim(ymin=0)
        #plt.grid()
        plt.title('Baseline: KL/Distances')
        plt.xlabel('KL', color=fig_color)
        plt.ylabel('Performance increase in relation to baseline', color=fig_color)
        plt.savefig(plotpath + '/correlation_inequality_kl_train_baseline.pdf', format='pdf', transparent=True)
        plt.close()

    # Gini
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(ginis, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(ginis, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': legend_size})
    plt.ylim(ymin=0)
    #plt.grid()
    plt.title('Gini/Distances')
    plt.xlabel('Gini', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_gini_train.pdf', format='pdf', transparent=True)
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

    plt.legend(loc=2, prop={'size': legend_size})
    plt.ylim(ymax=1)
    #plt.grid()
    plt.title('Lorenz curve: Stationary distributions')
    plt.xlabel('States', color=fig_color)
    plt.ylabel('Cumulative probability', color=fig_color)
    plt.savefig(plotpath + '/lorenz-curve.pdf', format='pdf', transparent=True)
    plt.close()

def norm_stationaries(estimated_stationaries):
    # Normalize and return
    return estimated_stationaries / np.sum(estimated_stationaries, axis=4)[:, :, :, :, np.newaxis]

def calc_stationariy_distances(est_norm):
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
    # runs, models, train steps, hammings

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

data = prepare_data(sources)  # runs, models, train steps, thresholds, h_ip, #ee-connections, test steps / test chunks

# Indices
#hpos_idx = np.where(para.c.h_ip_factor == 2.0)[0][0]  # index where h_ip = 2.0
hpos_idx = np.where(np.isclose(para.c.eta_ip, 0.001))[0][0]  # index where eta_ip = 0.001
mxthresh_idx = np.shape(data['transition_distances'])[3]-1  # index where hamming threshold is max
dens_idx = np.where(para.c.connections_density == 0.1)[0][0]  # index where connections_density = 0.1
train_idx = len(para.c.steps_plastic)-1  # index where training steps are max

# Prepare some data sets
normed_stationaries = norm_stationaries(data['estimated_stationaries'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:])
stationairy_distances = calc_stationariy_distances(normed_stationaries)

hamming_means = prepare_hamming(data['hamming_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:])
#ncomparison = data['ncomparison'][:,:,:,mxthresh_idx,hpos_idx,:]

#################### Training step plots ####################

# Plot transition performance for different training steps
if np.shape(data['transition_distances'])[2] > 1:
    training_steps_plot(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:],
        suffix="transition", ytext="Transition error")

# Plot stationary performance for different training steps
if np.shape(data['estimated_stationaries'])[2] > 1:
    training_steps_plot(stationairy_distances,
        suffix = "stationary", ytext = "Stationary error")

# Plot hamming mean for different training steps
if np.shape(data['hamming_distances'])[2] > 1:
    training_steps_plot(hamming_means,
        suffix="hamming", ytext="Relative mean hamming distance")

#################### Test chunk plots ####################

if len(para.c[ip]) > 1:
    y_max = np.max(np.array([np.max(data['transition_distances'][:, :, train_idx, mxthresh_idx, 0, dens_idx, :]),
                     np.max(data['transition_distances'][:, :, train_idx, mxthresh_idx, len(para.c[ip])-1, dens_idx, :])]))
    test_trace_plot(data['transition_distances'][:,:,:,mxthresh_idx,0,dens_idx,:],
                    suffix="distances_smallhip", ylabel="Transition error", title="small "+str(ip), ymax=0.5*y_max)
    test_trace_plot(data['transition_distances'][:, :, :, mxthresh_idx, len(para.c[ip])-1, dens_idx, :],
                    suffix="distances_hugehip", ylabel="Transition error", title="huge "+str(ip), ymax=0.5*y_max)

if len(para.c.connections_density) > 1:
    y_max = np.max(np.array([np.max(data['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, 0, :]),
                     np.max(data['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, len(para.c.connections_density)-1, :])]))

    for i in range(len(para.c.connections_density)):
        test_trace_plot(data['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, i, :],
                        suffix="distances_connectivity_train-max_"+str(i), ylabel="Transition error", title="connectivity: "+str(para.c.connections_density[i]), ymax=0.5*y_max)
    #for i in range(len(para.c.connections_density)):
    #    test_trace_plot(data['transition_distances'][:, :, 0, mxthresh_idx, hpos_idx, i, :],
    #                    suffix="distances_connectivity_train-min_" + str(i), ylabel="Transition error",
    #                    title="connectivity: " + str(para.c.connections_density[i]), ymax=0.5 * y_max)
    # test_trace_plot(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,0,:],
    #                 suffix="distances_sparse_connectivity", ylabel="Transition error", title="sparse connectivity", ymax=0.5*y_max)
    # test_trace_plot(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:],
    #                 suffix="distances_medium_connectivity", ylabel="Transition error", title="medium connectivity", ymax=0.5*y_max)
    # test_trace_plot(data['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, len(para.c.connections_density)-1, :],
    #                 suffix="distances_dense_connectivity", ylabel="Transition error", title="dense connectivity", ymax=0.5*y_max)

test_trace_plot(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:],
                suffix="distances", ylabel="Transition error")
test_trace_plot(stationairy_distances[:,:,:,:],
                suffix="stationary", ylabel="Stationary error")
#test_trace_plot(activity[:,:,:,mxthresh_idx,hpos_idx,:],
#                suffix="activity", label="Activity (percentage)")

#################### Hamming Distancs evaluation ####################

if np.shape(data['hamming_distances'])[3] > 1:
    hamming_histogram(data['hamming_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:])

if np.shape(data['transition_distances'])[2] > 1:
    training_steps_plot_thresholds(data['transition_distances'][:,:,:,mxthresh_idx,:,dens_idx,:])

#################### Activity/NComparison ####################

#activity_distance_correlation_plot(distances, activity)
#ncomparison_distance_correlation_plot(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,:], ncomparison[:,:,:,np.shape(ncomparison)[3]-1])

#################### Variance, Entropy, Gini / Lorenz ####################

inequality_distance_correlation_plot(data['transition_distances'][:,:,:,mxthresh_idx,:,dens_idx,:], hpos_idx)
lorenz_plot(normed_stationaries)

#################### weight strength ####################
#
# def flatten(T):
#     if type(T) != types.TupleType: return (T,)
#     elif len(T) == 0: return ()
#     else: return flatten(T[0]) + flatten(T[1:])
#
# # stationary [ 0.25 ,  0.375,  0.25 ,  0.125]
#
# max_train = np.shape(data['weights_ee'])[2]-1
# weights_ee = data['weights_ee'][:,:,max_train,mxthresh_idx,hpos_idx,:]
# weights_eu = data['weights_eu'][:,:,max_train,mxthresh_idx,hpos_idx,:]
# # now: runs / models / weight dense
#
# shape = np.shape(weights_ee)
#
# num_states = len(para.c.states)
# cluster_coefficient = np.zeros(flatten((np.shape(weights_ee)[0:3], (num_states,num_states))))
# num_connections = np.copy(cluster_coefficient)
# for g in range(shape[0]):
#     for h in range(shape[1]):
#         for k in range(shape[2]):
#             tmp_weights_eu = weights_eu[g, h, k, :, :]
#             tmp_weights_ee = weights_ee[g, h, k, :, :]
#
#             for i in range(num_states):
#                 for j in range(num_states):
#                     # Get weights of neurons for input state i and j
#                     # and make boolean out of it ('true' if neuron has input, 'false' if not)
#                     idx_i = tmp_weights_eu[:, i].astype(bool)
#                     idx_j = tmp_weights_eu[:, j].astype(bool)
#                     # Weights between neurons of state i and state j
#                     weights_ij = tmp_weights_ee[idx_i, :][:, idx_j]
#                     # Sum all weights to obtain coefficient
#                     cluster_coefficient[g, h, k, i, j] = np.sum(weights_ij)
#
#                     weights_ij[weights_ij < 0.0001] = 0
#                     num_connections[g, h, k, i, j] = np.sum(weights_ij.astype(bool))
#
# cluster_coefficient_mean = np.mean(cluster_coefficient, axis=0)
# num_connections_mean = np.mean(num_connections, axis=0)
#
# plt.imshow(num_connections_mean[0,0], origin='upper', cmap='copper', interpolation='none')
# plt.xticks(np.arange(num_states), para.c.states)
# plt.yticks(np.arange(num_states), para.c.states)
# plt.colorbar()
#
