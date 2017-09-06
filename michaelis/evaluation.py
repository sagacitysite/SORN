import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

# Parameters for evaluation
current = "2017-09-04_14-46-05"
test_step_size = 5000
train_step_size = 2500
train_offset = 5000
num_runs = 11 # How many runs should we evaluate

# Create path and get files
path = os.getcwd() + "/backup/test_multi/" + current
datapath = path + "/data"
plotpath = path + "/plots"

if not os.path.exists(plotpath):
    os.mkdir(plotpath)

files = glob.glob(os.path.join(datapath, "transition_distances_*"))

# Prepare data from files to numpy array
def prepare_data(files):
    global train_step_size, train_offset

    num_files = len(files)

    distances_raw = [dict() for x in range(num_files)]

    # Store file content in dictionary
    for i in range(num_files):
        run = int(files[i].split('_run')[1].split('_model')[0])
        model = int(files[i].split('_model')[1].split('_steps')[0])
        train_step = int(files[i].split('_steps')[1].split('.')[0])
        distances_raw[i] = {"run": run, "model": model, "train_step": train_step, "distance": np.load(files[i])}

    # Get sizes
    num_models = len(np.unique([dist['model'] for dist in distances_raw]))
    num_train_steps = len(np.unique([dist['train_step'] for dist in distances_raw]))
    min_num_test_steps = np.min([len(dist['distance']) for dist in distances_raw])

    # Prepare sorted numpy array
    distances = np.empty((num_runs, num_models, num_train_steps, min_num_test_steps))

    # Store dictionary data in clean numpy array
    for dist in distances_raw:
        for i in range(num_runs):
            for j in range(num_models):
                for k in range(num_train_steps):
                    if dist['run'] == i and dist['model'] == (j + 1) and dist['train_step'] == (train_offset + k * train_step_size):
                        distances[i,j,k] = dist['distance'][0:min_num_test_steps]
                        break
                    else:
                        continue

    # Return sorted and clean data
    return distances

# def get_distant_mean_of_model(distances_raw, search):
#     # Initalize arrays
#
#     steps = np.empty(num_train_steps)
#     distances = np.empty(num_train_steps)
#
#
#     i=0
#     for dist in distances_raw:
#         if dist['model'] == search:
#             steps[i] = dist['train_step']
#             # Mean is only appropriate if STDP is switched off in testing, otherwise take n-th value (where n is some position)
#             distances[i] = np.mean(dist['distance'])
#             i += 1
#
#     # Sort values
#     steps_sorted = np.sort(steps)
#     distances_sorted = distances[np.argsort(steps)]
#
#     return (steps_sorted, distances_sorted)

# def training_steps_plot(distances):
#     global plotpath
#
#     # Plot every model
#     num_models = len(np.unique([dist['model'] for dist in distances]))
#     for i in range(num_models):
#         (steps, dists) = get_distant_mean_of_model(distances, i + 1)
#         legend = 'Model '+str(i+1)
#         plt.plot(steps, dists, label=legend)
#
#     # Beautify plot and save png file
#     plt.legend()
#     plt.ylim(ymin=0)
#     plt.xlabel('Training steps')
#     plt.ylabel('Mean squared distance to initial transition')
#     plt.savefig(plotpath + '/distances_training_steps.png')
#     plt.close()

def training_steps_plot(distances):
    global plotpath

    # Get mean over "runs" and over "train_steps"
    # CAUTION: Mean over "train_steps" is only appropriate if STDP is switched off in test phase
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
        (_, caps, _) = plt.errorbar(train_steps, dists_mean[i,:], label=legend, yerr=dists_std[i], color=color_palette[i])

        for cap in caps:
            cap.set_markeredgewidth(0.1)

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

def test_trace_plot(distances):
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
        plt.errorbar(test_steps, dists_mean[i], label=legend, yerr=dists_std[i], color=color_palette[i])

        # Plot all runs of every model (transparent in background)
        for j in range(np.shape(dists)[0]):
            plt.plot(test_steps, dists[j,i], color=color_palette[i], alpha=0.1)

    # Beautify plot and save png file
    plt.legend()
    plt.ylim(ymin=0)
    plt.xlabel('Test steps')
    plt.ylabel('Mean squared distance to initial transition')
    plt.savefig(plotpath + '/distances_test_traces.png', dpi=144)
    plt.close()

distances = prepare_data(files) # (runs, models, train steps, test steps)

training_steps_plot(distances)
#test_trace_plot(distances)
