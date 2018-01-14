import numpy as np
import os
import glob
import sys
import types
import scipy
from scipy.stats import pearsonr

# Import and initalize matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

#rcParams.keys()

fig_color = '#000000'
legend_size = 12  # 10

rcParams['font.family'] = 'CMU Serif'
rcParams['font.size'] = '15'  # 14/20
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
num_runs = 10  # How many runs should we evaluate

# Prepare path and get files
os.chdir("../backup/test_size/" + str(sys.argv[1]))  #"2017-12-31_12-01-55_models5"
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
utils.backup = backup_overwrite  # Overwrite utils backup function to avoid new backup copies

# Import parameters
#import param_mcmc_multi as para
import imp
para = imp.load_source('param_mcmc_multi', path+'/michaelis/param_mcmc_size.py')
#import utils.stationary as stationaryDistribution


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

def impute_nan(dat, i, j):
    min_val, max_val = None, None
    
    k = 1
    while True:
        # If index exists, value is not nan and min/max value was not already set
        if i-k >= 0 and not np.isnan(dat[i-k,j]) and min_val is None:
            min_val = dat[i-k,j]
        if i+k < len(dat[:,j]) and not np.isnan(dat[i+k,j]) and max_val is None:
            max_val = dat[i+k,j]
        
        # If counter is out of both indices, break loop
        if i-k < 0 or i+k >= len(dat[:,j]):
            break
        else:
            k += 1
            
    # If no min/max val was found
    if min_val is not None and max_val is not None:
        return (max_val + min_val)/2
    elif min_val is None and max_val is not None:
        return max_val
    elif min_val is not None and max_val is None:
        return min_val
    elif min_val is None and max_val is None:
        raise Exception('Imputation was not possible, too many values are missing.')
    
def impute(dat):
    num_neurons = len(para.c.N_e)
    num_inputs = len(para.c.N_u_e_coverage)
    
    for i in range(num_neurons):
        for j in range(num_inputs):
            if np.isnan(dat[i,j]):
                dat[i,j] = impute_nan(dat, i, j)
                
    return dat
    

def plot_2d(distances):
    global plotpath

    neurons_array = para.c.N_e
    inputs_array = para.c.N_u_e_coverage
    
    num_models = np.shape(distances)[1]
    num_test = np.shape(distances)[4]
    
    # Get get last test chunk and mean over runs
    dists_mean = np.mean(distances[:,:,:,:,num_test-1], axis=0)
    # dists_mean: models, num_neurons, num_input
    
    # Impute nan/missing values
    dists_mean = np.array([impute(dists_mean[i,:,:]) for i in range(num_models)])

    # Prepare data for surface plot
    max_dist = np.max(dists_mean)
    X, Y = np.meshgrid(inputs_array, neurons_array)
    
    colors = np.array(cm.coolwarm(np.linspace(0,1,256)))
    colors[:,3] = 0.9
    cmap = LinearSegmentedColormap.from_list('alpha_cmap', colors.tolist())
    
    for i in range(num_models):
        # Plot the surface
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, dists_mean[i,:,:], cmap=cmap, linewidth=0, antialiased=True,
                               vmin=0, vmax=max_dist-0.05, rstride=1, cstride=1)

        # Customize
        ax.set_zlim(0, max_dist)
        ttl = plt.title('Model ' + str(i+1))
        #ttl.set_position([.5, 1.05])
        ttl.set_position([0.5, 0.96])  # [horizontal, vertical]

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Store in file
        plt.savefig(plotpath + '/distances_size_inputs_' + str(i) + '.png', dpi=144)
        plt.close()


##################################################
#################### Evaluate ####################
##################################################

#################### Prepare data ####################

data = prepare_data(sources)  # runs, models, num_neurons, num_input, train chunks

plot_2d(data['transition_distances'])

