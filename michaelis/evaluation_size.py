import numpy as np
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

# Parameters for evaluation
current = "2017-09-21_15-14-08"
test_step_size = 5000
network_size = np.arange(100,501,50)
input_perc = np.arange(0.03,0.151,0.01)
num_runs = np.array([0]) # How many runs should we evaluate
# Bad: 8-14

# Create path and get files
path = os.getcwd() + "/backup/test_size/" + current
datapath = path + "/data"
plotpath = path + "/plots"

if not os.path.exists(plotpath):
    os.mkdir(plotpath)

files = glob.glob(os.path.join(datapath, "transition_distances_*"))

# Prepare data from files to numpy array
def prepare_data(files):
    global num_runs, network_size, input_perc

    num_files = len(files)

    #distances_raw = [dict() for x in range(num_files)]
    distances_raw = []

    # Store file content in dictionary
    for i in range(num_files):
        run = int(files[i].split('_run')[1].split('_model')[0])
        model = int(files[i].split('_model')[1].split('_neurons')[0])
        neurons = int(files[i].split('_neurons')[1].split('_input')[0])
        input = int(files[i].split('_input')[1].split('.')[0])
        if (run in num_runs) and (neurons in np.arange(len(network_size))) and  (input in np.arange(len(input_perc))):
            distances_raw.append({"run": run, "model": model, "neurons": neurons, "input": input, "distance": np.nan_to_num(np.load(files[i]))})
        #if (run in num_runs) and (neurons in network_size) and (input in np.floor(input_perc*neurons)):
            #distances_raw[i] = {"run": run, "model": model, "neurons": neurons, "input": input, "distance": np.load(files[i])}
            #distances_raw.append({"run": run, "model": model, "neurons": neurons, "input": input, "distance": np.load(files[i])})

    # Get sizes
    num_models = len(np.unique([dist['model'] for dist in distances_raw]))
    #num_neurons = len(np.unique([dist['neurons'] for dist in distances_raw]))
    num_neurons = len(network_size)
    #num_input = len(np.unique([dist['input'] for dist in distances_raw])) ## This is not correct!!
    num_input = len(input_perc)
    min_num_test_steps = np.min([len(dist['distance']) for dist in distances_raw])

    #for dist in distances_raw:
    #    perc = np.ceil(100*dist['input']/dist['neurons'])/100
    #    #print(perc)
    #    if perc == 0.09:
    #        print('Model: ' + str(dist['model']) + ', Neurons: ' + str(dist['neurons']) + ', Input: ')

    # Prepare sorted numpy array
    distances = np.empty((len(num_runs), num_models, num_neurons, num_input, min_num_test_steps))

    # Store dictionary data in clean numpy array
    for dist in distances_raw:
        for i in range(len(num_runs)):
            for j in range(num_models):
                for k in range(num_neurons):
                    for l in range(num_input):
                        #cur_input_size = np.floor(input_perc[l]*network_size[k])
                        #if dist['run'] == i and dist['model'] == j and dist['neurons'] == network_size[k] and dist['input'] == cur_input_size:
                        if dist['run'] == i and dist['model'] == j and dist['neurons'] == k and dist['input'] == l:
                            distances[i,j,k,l] = dist['distance'][0:min_num_test_steps]
                            break
                        else:
                            continue

    # Return sorted and clean data
    return distances

def plot_2d(distances):
    global plotpath, network_size, input_perc

    # Get mean over "runs" and over "train_steps"
    # CAUTION: Mean over "test_steps" is only appropriate if STDP is switched off in test phase
    dists_mean = np.mean(distances, axis=(0, 4))
    dists_mean[dists_mean < 0] = 0 # correct some numeric variations

    dists_mean[dists_mean > 0.3] = 0.3 # FIXME

    X, Y = np.meshgrid(input_perc, network_size)

    num_models = np.shape(distances)[1]
    for i in range(num_models):
        # Plot the surface
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, dists_mean[i,:,:], cmap=cm.coolwarm, linewidth=0, antialiased=False, rstride=1, cstride=1)

        # Customize
        ax.set_zlim(np.min(dists_mean), np.max(dists_mean))
        plt.title('Model ' + str(i+1))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Store in file
        plt.savefig(plotpath + '/distances_size_inputs_' + str(i) + '.png', dpi=144)
        plt.close()

distances = prepare_data(files) # (runs, models, neurons, input, test steps)

plot_2d(distances)

