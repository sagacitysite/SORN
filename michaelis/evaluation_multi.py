import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import scipy
from scipy.stats import pearsonr

# Parameters for evaluation
current = "2017-10-06_14-48-05"
test_step_size = 5000
train_step_size = 5000
train_offset = 5000
num_runs = 20 # How many runs should we evaluate

# Create path and get files
path = os.getcwd() + "/backup/test_multi/" + current
datapath = path + "/data"
plotpath = path + "/plots"

if not os.path.exists(plotpath):
    os.mkdir(plotpath)

files_distances = glob.glob(os.path.join(datapath, "transition_distances_*"))
files_activity = glob.glob(os.path.join(datapath, "activity_*"))
files_ncomparison = glob.glob(os.path.join(datapath, "hamming_input_data_*"))

files = {'distance': files_distances, 'activity': files_activity, 'ncomparison': files_ncomparison}

# Prepare data from files to numpy array
def prepare_data(files):
    global train_step_size, train_offset

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

    # Store file content in dictionary
    for i in range(num_files):
        run = int(files['distance'][i].split('_run')[1].split('_model')[0])
        model = int(files['distance'][i].split('_model')[1].split('_steps')[0])
        train_step = int(files['distance'][i].split('_steps')[1].split('.')[0])
        distances_raw[i] = {"run": run, "model": model, "train_step": train_step,
                            "distance": np.load(files['distance'][i]),
                            "activity": np.load(files['activity'][i]),
                            "ncomparison": np.load(files['ncomparison'][i]) if ncomp_exists else None }

    # Get sizes
    num_models = len(np.unique([dist['model'] for dist in distances_raw]))
    num_train_steps = len(np.unique([dist['train_step'] for dist in distances_raw]))
    num_test_steps = np.unique([len(dist['distance']) for dist in distances_raw])[0]

    # Prepare sorted numpy array
    distances = np.empty((num_runs, num_models, num_train_steps, num_test_steps))
    activity = np.copy(distances)

    if ncomp_exists:
        ncomparison = np.empty((num_runs, num_models, num_train_steps))
    else:
        ncomparison = None

    # Store dictionary data in clean numpy array
    for dist in distances_raw:
        for i in range(num_runs):
            for j in range(num_models):
                for k in range(num_train_steps):
                    #if dist['run'] == i and dist['model'] == (j + 1) and dist['train_step'] == (train_offset + k * train_step_size):
                    if dist['run'] == i and dist['model'] == j and dist['train_step'] == k:
                        distances[i,j,k] = dist['distance'][0:num_test_steps]
                        activity[i,j,k] = dist['activity'][0:num_test_steps]

                        if ncomp_exists:
                            ncomparison[i, j, k] = dist['ncomparison']
                        break
                    else:
                        continue

    # Return sorted and clean data
    return (distances, activity, ncomparison)

def training_steps_plot(distances):
    global plotpath

    # Get mean over "runs" and over "train_steps"
    # CAUTION: Mean over "test_steps" is only appropriate if STDP is switched off in test phase
    dists_mean = np.mean(distances, axis=(0,3))
    dists_std = np.std(distances, axis=(0,3))

    # Mean just over test phase, not over runs
    dists_mean_single = np.mean(distances, axis=3)

    # Get number of models and calculate train steps
    num_models = np.shape(dists_mean)[0]
    train_steps = train_offset + np.arange(np.shape(dists_mean)[1])*train_step_size

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, num_models))

    # Plot influence of training steps for every model
    for i in range(num_models):
        legend = 'Model '+str(i+1)
        plt.errorbar(train_steps, dists_mean[i,:], label=legend, yerr=dists_std[i], color=color_palette[i],
                     elinewidth=1, ecolor=np.append(color_palette[i][0:3], 0.5))

        # Plot all runs of every model (transparent in background)
        for j in range(np.shape(dists_mean_single)[0]):
            plt.plot(train_steps, dists_mean_single[j, i], color=color_palette[i], alpha=0.1)

    # Beautify plot and save png file
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
        x = np.mean(dists[:, i, :], axis=1)
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


(distances, activity, ncomparison) = prepare_data(files) # (runs, models, train steps, test steps)

# Plot performance for different testing conditions
if np.shape(distances)[2] > 1:
    training_steps_plot(distances)

# Plot performance and activity in testing phase
test_trace_plot(distances, prefix="distances", label="Mean squared distance to initial transition")
test_trace_plot(activity, prefix="activity", label="Activity (percentage)")

# Plot correlation between performance and activity/ncomparison
activity_distance_correlation_plot(distances, activity)
if not (ncomparison is None):
    ncomparison_distance_correlation_plot(distances, ncomparison)
