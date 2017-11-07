import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import scipy
from scipy.stats import pearsonr

# Path and num runs value for evaluation
current = "2017-11-03_13-18-54_hamming"
num_runs = 10 # How many runs should we evaluate

# Create path and get files
path = os.getcwd() + "/backup/test_multi/" + current
datapath = path + "/data"
plotpath = path + "/plots"

if not os.path.exists(plotpath):
    os.mkdir(plotpath)

files_distances = glob.glob(os.path.join(datapath, "transition_distances_*"))
files_activity = glob.glob(os.path.join(datapath, "activity_*"))
files_ncomparison = glob.glob(os.path.join(datapath, "hamming_input_data_*"))
files_hamming = glob.glob(os.path.join(datapath, "hamming_distances_*"))
files_stationary = glob.glob(os.path.join(datapath, "stationairies_*")) # TODO change to stationaries

files = {'distance': files_distances, 'activity': files_activity, 'stationary'; stationary, 'ncomparison': files_ncomparison, 'hamming': files_hamming}

# Import parameters
sys.path.insert(0,path+"/michaelis")
sys.path.insert(0,os.getcwd()+"/utils")
import param_mcmc_multi as para
import stationairy as stationairyDistribution

# Parameters for evaluation
num_states = np.shape(para.c.source.transitions)[1]
test_step_size = para.c.stats.transition_step_size
test_steps = para.c.steps_noplastic_test
steps_plastic = para.c.steps_plastic
train_step_size = para.c.steps_plastic[1]-para.c.steps_plastic[0] if np.size(para.c.steps_plastic) > 1 else 0
train_offset = np.min(para.c.steps_plastic)

# Prepare data from files to numpy array
def prepare_data(files):
    global test_steps, train_step_size, train_offset

    num_files_dist = len(files['distance'])
    num_files_act = len(files['activity'])
    num_files = num_files_dist

    # Check if we have equal number of files
    if num_files_dist != num_files_act:
        raise Exception(
            'Number of files differ... something went wrong!')

    distances_raw = [dict() for x in range(num_files)]

    # Check if ncomparison is part of the file structure
    ncomp_exists = True
    if not files_ncomparison:
        ncomp_exists = False

    # Check if hamming is part of the file structure
    hamming_exists = True
    if not files_hamming:
        hamming_exists = False

    print('# Store files in dictionary')

    # Store file content in dictionary
    for i in range(num_files):
        run = int(files['distance'][i].split('_run')[1].split('_model')[0])
        model = int(files['distance'][i].split('_model')[1].split('_threshold')[0])
        threshold = int(files['distance'][i].split('_threshold')[1].split('_steps')[0])
        train_step = int(files['distance'][i].split('_steps')[1].split('.')[0])
        distances_raw[i] = {"run": run, "model": model, "train_step": train_step, "threshold": threshold,
                            "distance": np.load(files['distance'][i]),
                            "activity": np.load(files['activity'][i]),
                            "stationary": np.load(files['stationary'][i]),
                            "ncomparison": np.load(files['ncomparison'][i]) if ncomp_exists else None,
                            "hamming": np.load(files['hamming'][i]) if hamming_exists else None }

    # Get sizes
    num_models = len(np.unique([dist['model'] for dist in distances_raw]))
    num_train_steps = len(np.unique([dist['train_step'] for dist in distances_raw]))
    num_test_chunks = np.unique([len(dist['distance']) for dist in distances_raw])[0]
    num_thresholds = len(np.unique([dist['threshold'] for dist in distances_raw]))

    # Prepare sorted numpy array
    distances = np.empty((num_runs, num_models, num_train_steps, num_thresholds, num_test_chunks))
    activity = np.copy(distances)
    stationary = np.copy(distances)
    ncomparison = np.empty((num_runs, num_models, num_train_steps, num_thresholds)) if ncomp_exists else None
    hamming = np.empty((num_runs, num_models, num_train_steps, num_thresholds, test_steps)) if hamming_exists else None

    print('# Store dictionary in arrays')
    # Store dictionary data in clean numpy array
    for dist in distances_raw:
        for i in range(num_runs):
            for j in range(num_models):
                for k in range(num_train_steps):
                    for h in range(num_thresholds):
                        #if dist['run'] == i and dist['model'] == (j + 1) and dist['train_step'] == (train_offset + k * train_step_size):
                        if dist['run'] == i and dist['model'] == j and dist['train_step'] == k and dist['threshold'] == h:
                            distances[i,j,k,h] = dist['distance'][0:num_test_chunks]
                            activity[i,j,k,h] = dist['activity'][0:num_test_chunks]
                            stationary[i,j,k,h] = dist['stationary'][0:num_test_chunks]

                            if ncomp_exists:
                                ncomparison[i,j,k,h] = dist['ncomparison']

                            if hamming_exists:
                                hamming[i,j,k,h] = dist['hamming']
                            break
                        else:
                            continue

    # Return sorted and clean data
    return (distances, activity, stationary, ncomparison, hamming)

def training_steps_plot(distances):
    global plotpath, steps_plastic

    # Get mean over "runs" and over "train_steps"
    # CAUTION: Mean over "test_steps" is only appropriate if STDP is switched off in test phase
    dists_mean = np.mean(distances, axis=(0,3))
    dists_std = np.std(distances, axis=(0,3))

    # Mean just over test phase, not over runs
    dists_mean_single = np.mean(distances, axis=3)

    # Get number of models and calculate train steps
    num_models = np.shape(dists_mean)[0]

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, num_models))

    # Plot influence of training steps for every model
    for i in range(num_models):
        legend = 'Model '+str(i+1)
        plt.errorbar(steps_plastic, dists_mean[i,:], label=legend, yerr=dists_std[i], color=color_palette[i],
                     elinewidth=1, ecolor=np.append(color_palette[i][0:3], 0.5))

        # Plot all runs of every model (transparent in background)
        for j in range(np.shape(dists_mean_single)[0]):
            plt.plot(steps_plastic, dists_mean_single[j, i], color=color_palette[i], alpha=0.1)

    # Beautify plot and save png file
    plt.legend()
    plt.ylim(ymin=0)
    plt.xlabel('Training steps')
    plt.ylabel('Mean squared distance to initial transition')
    plt.savefig(plotpath + '/distances_training_steps.png', dpi=144)
    plt.close()

def training_steps_plot_hamming(hamming):
    global plotpath, steps_plastic

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
        plt.errorbar(steps_plastic, h_mean[model, :, i], label=legend, yerr=h_std[i], color=color_palette[i],
                     elinewidth=1, ecolor=np.append(color_palette[i][0:3], 0.5))

    # Beautify plot and save png file
    plt.title('Model ' + str(model))
    plt.legend()
    plt.ylim(ymin=0)
    plt.xlabel('Training steps')
    plt.ylabel('Mean squared distance to initial transition')
    plt.savefig(plotpath + '/distances_training_steps.png', dpi=144)
    plt.close()

def test_trace_plot(distances, prefix, label):
    global plotpath, test_step_size

    # Get results for highest train step only
    last_idx_train_steps = np.shape(distances)[2]-1
    dists = distances[:,:,last_idx_train_steps,:]

    # Calculate mean and standard deviation
    dists_mean = np.mean(dists, axis=0)
    dists_std = np.std(dists, axis=0)

    # Get number of original test steps (for x axis)
    test_steps = np.arange(np.shape(dists)[2]) * test_step_size

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
    plt.savefig(plotpath + '/' + prefix + '_test_traces.png', dpi=144)
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
    global plotpath, train_step_size, train_offset

    # Calculate stationairy distributions from markov chains
    stationaries = [stationairyDistribution.calculate(transition) for transition in para.c.source.transitions]
    variance = np.var(stationaries, axis=1)  # TODO maybe choose some other measure (additionally to variance and entropy)
    entropy = [scipy.stats.entropy(s, np.repeat(0.25, num_states)) for s in stationaries]

    # Get number of train steps
    train_steps = np.shape(distances)[2]

    # Exclude first test step and mean over test steps
    dists = np.mean(distances[:, :, :, 1:], axis=3)

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, train_steps))

    for i in range(train_steps):
        # Variance
        legend = str(train_offset + train_step_size*i) + ' training steps, r=' + str(np.round(pearsonr(variance, np.mean(dists[:,:,i], axis=0))[0],2))
        plt.errorbar(variance, np.mean(dists[:,:,i], axis=0), label=legend, yerr=np.std(dists[:,:,i], axis=0),# fmt='o',
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
            legend = str(train_offset + train_step_size * i) + ' training steps, r=' + str(np.round(pearsonr(variance, diff)[0], 2))
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

(distances, activity, stationary, ncomparison, hamming) = prepare_data(files) # (runs, models, train steps, thresholds, test steps / test chunks)

# Plot performance for different training steps
if np.shape(distances)[2] > 1:
    training_steps_plot(get_max_threshold(distances))

# Plot performance for different training steps
if np.shape(distances)[2] > 1:
    training_steps_plot(get_max_threshold(distances))

# Plot hamming mean for different training steps
if np.shape(hamming)[2] > 1:
    training_steps_plot_hamming(hamming)

# Plot performance and activity in testing phase
test_trace_plot(get_max_threshold(distances), prefix="distances", label="Mean squared distance to initial transition")
#test_trace_plot(get_max_threshold(activity), prefix="activity", label="Activity (percentage)")

# Plot correlation between performance and activity/ncomparison
#activity_distance_correlation_plot(distances, activity)
if not (ncomparison is None):
    ncomparison_distance_correlation_plot(get_max_threshold(distances), ncomparison[:,:,:,np.shape(ncomparison)[3]-1])

inequality_distance_correlation_plot(get_max_threshold(distances)) # entropy plot
