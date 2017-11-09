import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import scipy
from scipy.stats import pearsonr

# Path and num runs value for evaluation
current = "2017-11-09_00-07-53"
num_runs = 9 # How many runs should we evaluate

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

# Parameters for evaluation
#num_states = np.shape(para.c.source.transitions)[1]
#test_step_size = para.c.stats.transition_step_size
#steps_plastic = para.c.steps_plastic
#train_step_size = para.c.steps_plastic[1]-para.c.steps_plastic[0] if np.size(para.c.steps_plastic) > 1 else 0
#train_offset = np.min(para.c.steps_plastic)

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

def training_steps_plot_hamming(hamming):
    global plotpath

    tt = np.delete(hamming, 5, 0)

    ttt = [np.nanmax(tt[:,:,:,:,i]) for i in range(np.shape(hamming)[4]-1) ]

    # Get mean over "runs" and "training steps" (hamming distance for every training step)
    h_mean = np.mean(hamming[:,:,:,:,], axis=0) # results in [models, training steps, thresholds]
    h_std = np.std(hamming, axis=0)

    # Get number of models and calculate train steps
    num_models = np.shape(h_mean)[0]
    num_thresholds = np.shape(h_mean)[2]

    thresholds = para.c.stats.hamming_threshold

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, num_models))
    model = 0

    # Plot influence of training steps for every model
    for i in range(num_thresholds):
        legend = 'Threshold ' + str(thresholds[i])
        plt.errorbar(para.c.steps_plastic, h_mean[model, :, i], label=legend, yerr=h_std[i], color=color_palette[i],
                     elinewidth=1, ecolor=np.append(color_palette[i][0:3], 0.5))

    # Beautify plot and save png file
    plt.title('Model ' + str(model))
    plt.legend()
    plt.ylim(ymin=0)
    plt.xlabel('Training steps')
    plt.ylabel('Mean squared distance to initial transition')
    plt.savefig(plotpath + '/distances_training_steps.png', dpi=144)
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

def inequality_distance_correlation_plot(distances):
    global plotpath, para

    # Calculate stationary distributions from markov chains
    stationaries = [stationaryDistribution.calculate(transition) for transition in para.c.source.transitions]
    variance = np.var(stationaries, axis=1)  # TODO maybe choose some other measure (additionally to variance and entropy)
    entropy = [scipy.stats.entropy(s, np.repeat(0.25, np.shape(para.c.source.transitions)[1])) for s in stationaries]

    # Get number of train steps
    train_steps = np.shape(distances)[2]

    # Exclude first test step and mean over test steps
    dists = np.mean(distances[:, :, :, 1:], axis=3)

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, train_steps))

    for i in range(train_steps):
        # Variance
        legend = str(para.c.steps_plastic) + ' training steps, r=' + str(np.round(pearsonr(variance, np.mean(dists[:,:,i], axis=0))[0],2))
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

    # Dist baseline plot
    dists_baseline = np.mean(dists[:,:,0], axis=0)
    for i in range(train_steps):
        if i > 0:
            # Variance with distance difference
            diff = dists_baseline - np.mean(dists[:, :, i], axis=0)
            legend = str(para.c.steps_plastic) + ' training steps, r=' + str(np.round(pearsonr(variance, diff)[0], 2))
            plt.plot(variance, diff, label=legend, color=color_palette[i])

    plt.legend(prop={'size': 6})
    #plt.ylim(ymin=0)
    plt.grid()
    plt.title('Baseline: Variance/Distances')
    plt.xlabel('Variance')
    plt.ylabel('Performance increase in relation to baseline')
    plt.savefig(plotpath + '/correlation_inequality_variance_baseline.png', dpi=144)
    plt.close()

    # plt.scatter(variance, np.std(dists, axis=0))
    # plt.title('Correlation Variance / Deviation of distances (%.2f)' % pearsonr(variance, np.std(dists, axis=0))[0])
    # plt.xlabel('Variance')
    # plt.ylabel('Deviation of mean squared distance to initial transition')
    # plt.ylim(ymin=0)
    # plt.savefig(plotpath + '/correlation_inequality_variance_std.png', dpi=144)
    # plt.close()

    # Entropy
    # plt.errorbar(entropy, np.mean(dists, axis=0), yerr=np.std(dists, axis=0), fmt='o')
    # plt.title('Correlation Entropy/Distances (%.2f)' % pearsonr(entropy, np.mean(dists, axis=0))[0])
    # plt.xlabel('Entropy')
    # plt.ylabel('Mean squared distance to initial transition')
    # plt.savefig(plotpath + '/correlation_inequality_entropy.png', dpi=144)
    # plt.close()
    #
    # plt.scatter(entropy, np.std(dists, axis=0))
    # plt.title('Correlation Entropy / Deviation of distances (%.2f)' % pearsonr(entropy, np.std(dists, axis=0))[0])
    # plt.xlabel('Entropy')
    # plt.ylabel('Deviation of mean squared distance to initial transition')
    # plt.ylim(ymin=0)
    # plt.savefig(plotpath + '/correlation_inequality_entropy_std.png', dpi=144)
    # plt.close()

def get_max_threshold(arr):
    return arr[:,:,:,np.shape(arr)[3]-1,:]

def prepare_stationary(estimated_stationaries):
    global para

    # Calculate stationaries
    T = para.c.source.transitions
    stationaries = np.stack([ stationaryDistribution.calculate(trans) for trans in T])

    # Normalize
    est_norm = estimated_stationaries/np.sum(estimated_stationaries, axis=4)[:,:,:,:,np.newaxis]

    # Calculate distances
    # runs, models, train steps, thresholds, test chunks, stationary
    diff = est_norm - stationaries[np.newaxis, :, np.newaxis, np.newaxis, :]
    distances = np.sum(np.square(diff), axis=4)

    return distances

##################################################
#################### Evaluate ####################
##################################################

# transition_distances, activity, estimated_stationaries, ncomparison, hamming_distances

# Stationary: Estimatated stationaries
data = prepare_data(sources)  # (runs, models, train steps, thresholds, test steps / test chunks)
# Prepare stationary distances
stationairy_distances =  prepare_stationary(get_max_threshold(data['estimated_stationaries']))

# Plot transition performance for different training steps
if np.shape(data['transition_distances'])[2] > 1:
    training_steps_plot(get_max_threshold(data['transition_distances']),
        suffix="transition", ytext="Mean squared distance to initial transition")

# Plot stationary performance for different training steps
if np.shape(data['estimated_stationaries'])[2] > 1:
    training_steps_plot(stationairy_distances,
        suffix = "stationary", ytext = "Mean squared distance to stationary")

# Plot hamming mean for different training steps
#if np.shape(data['hamming_distances'])[2] > 1:
#    training_steps_plot_hamming(data['hamming_distances'])

# Plot performance and activity in testing phase
test_trace_plot(get_max_threshold(data['transition_distances']), suffix="distances", label="Mean squared distance to initial transition")
test_trace_plot(stationairy_distances, suffix="stationary", label="Mean squared distance to stationary")
#test_trace_plot(get_max_threshold(activity), prefix="activity", label="Activity (percentage)")

# Plot correlation between performance and activity/ncomparison
#activity_distance_correlation_plot(distances, activity)
#ncomparison = get_max_threshold(data['transition_distances']
#ncomparison_distance_correlation_plot(get_max_threshold(data['transition_distances']), ncomparison[:,:,:,np.shape(ncomparison)[3]-1])

inequality_distance_correlation_plot(get_max_threshold(data['transition_distances']))  # entropy plot

