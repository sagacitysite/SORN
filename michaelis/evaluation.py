import numpy as np
import os
import glob
import matplotlib.pyplot as plt

current = "2017-08-31_12-48-13"
path = os.getcwd() + "/backup/test_multi/" + current
datapath = path + "/data"
plotpath = path + "/plots"

if not os.path.exists(plotpath):
    os.mkdir(plotpath)

files = glob.glob(os.path.join(datapath, "transition_distances_*"))

def prepare_data(files):
    num_files = len(files)

    train_steps = np.empty(num_files)
    models = np.empty(num_files)
    distances = [dict() for x in range(num_files)]

    for i in range(num_files):
        models[i] = int(files[i].split('_model')[1].split('_steps')[0])
        train_steps[i] = int(files[i].split('_steps')[1].split('.')[0])
        distances[i] = {"model": models[i], "train_step": train_steps[i], "distance": np.load(files[i])}

    return distances

def get_distant_mean_of_model(distances_all, search):
    # Initalize arrays
    num_train_steps = len(np.unique([dist['train_step'] for dist in distances_all]))
    steps = np.empty(num_train_steps)
    distances = np.empty(num_train_steps)

    i=0
    for dist in distances_all:
        if dist['model'] == search:
            steps[i] = dist['train_step']
            # Mean is only appropriate if STDP is switched off in testing, otherwise take n-th value (where n is some position)
            distances[i] = np.mean(dist['distance'])
            i += 1

    # Sort values
    steps_sorted = np.sort(steps)
    distances_sorted = distances[np.argsort(steps)]

    return (steps_sorted, distances_sorted)

def compare_models_plot(distances):
    global plotpath

    # Plot every model
    num_models = len(np.unique([dist['model'] for dist in distances]))
    for i in range(num_models):
        (steps, dists) = get_distant_mean_of_model(distances, i + 1)
        legend = 'Model '+str(i+1)
        plt.plot(steps, dists, label=legend)

    # Beautify plot and save png file
    plt.legend()
    plt.xlabel('Training steps')
    plt.ylabel('Mean squared distance to initial transition')
    plt.savefig(plotpath + '/distances_training_steps.png')
    plt.close()

def get_last_train_step(distances_all):
    max_train_step = np.max([dist['train_step'] for dist in distances_all])

    models = []
    distances = []

    for dist in distances_all:
        if dist['train_step'] == max_train_step:
            models.append(dist['model'])
            distances.append(dist['distance'])

    # Convert list to numpy array
    models = np.asarray(models)
    distances = np.asarray(distances)

    # Sort values
    models_sorted = np.sort(models)
    distances_sorted = distances[np.argsort(models)]

    return (models_sorted, distances_sorted)

def distance_trace_plot(distances, test_step_size):
    global plotpath

    (models, dists) = get_last_train_step(distances)
    test_steps = (np.arange(np.shape(dists)[1])+1)*test_step_size

    for i in range(len(models)):
        legend = 'Model '+str(i+1)
        plt.plot(test_steps, dists[i], label=legend)

    # Beautify plot and save png file
    plt.legend()
    plt.xlabel('Test steps')
    plt.ylabel('Mean squared distance to initial transition')
    plt.savefig(plotpath + '/distances_test_traces.png')
    plt.close()

distances = prepare_data(files)

compare_models_plot(distances)
distance_trace_plot(distances, 5000)