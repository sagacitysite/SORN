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
legend_size = 12  # 12/14

rcParams['font.family'] = 'CMU Serif'
rcParams['font.size'] = '14'  # 14/20
rcParams['text.color'] = fig_color
rcParams['axes.edgecolor'] = fig_color
rcParams['xtick.color'] = fig_color
rcParams['ytick.color'] = fig_color
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = ':'
rcParams['grid.linewidth'] = 0.5
rcParams['grid.color'] = fig_color
rcParams['legend.fancybox'] = True
rcParams['legend.framealpha'] = 0.75
rcParams['patch.linewidth'] = 0

# Path and num runs value for evaluation
num_runs = 20  # How many runs should we evaluate

# Prepare path and get files
os.chdir("../backup/test_noplastic_train/" + str(sys.argv[1]))  #"2017-12-31_12-01-55_models5"
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
para = imp.load_source('param_mcmc_noplastic_train', path+'/michaelis/param_mcmc_noplastic_train.py')
#import utils.stationary as stationaryDistribution


sources = { 'transition_distances': None, 'estimated_stationaries': None }

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

def noplastic_train(distances):
    # distances: runs, models, noplastic train, train chunks
    last_chunk = np.shape(distances)[3]-1
    num_noplastic = np.shape(distances)[2]
    num_models = np.shape(distances)[1]
    
    dists_mean = np.mean(distances[:,:,:,last_chunk], axis=0)
    dists_std = np.std(distances[:,:,:,last_chunk], axis=0)
    # dists: models, noplastic train
    
    cols = cm.rainbow(np.linspace(0, 1, num_noplastic))
    
    for i in range(num_noplastic):
        legend = str('T_noplastic = ' + str(para.c.steps_noplastic_train[i]))
        plt.errorbar(np.arange(num_models)+1, dists_mean[:,i], label=legend, yerr=dists_std[:,i], marker='o',# fmt='o',
                     color=cols[i], ecolor=np.append(cols[i][0:3], 0.5))

    plt.legend(prop={'size': legend_size})
    plt.xlim(xmin=0.5, xmax=num_models+0.5)
    plt.ylim(ymin=0)
    plt.xlabel('Model', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/noplastic_train.pdf', format='pdf', transparent=True)
    plt.close()
    
    # t-Tests
    file = ""
    for i in range(num_noplastic):
        for j in range(num_noplastic):
            if j > i:
                res =scipy.stats.ttest_ind(distances[:,0,i,last_chunk], distances[:,0,j,last_chunk])
                file += "noplastic "+str(para.c.steps_noplastic_train[i])+" vs. "+str(para.c.steps_noplastic_train[j])+": t="+str(res[0])+", p="+str(res[1])+"\n"
    text_file = open(plotpath + "/t-tests_noplastic.txt", "w")
    text_file.write(file)
    text_file.close()

##################################################
#################### Evaluate ####################
##################################################

#################### Prepare data ####################

data = prepare_data(sources)  # runs, models, noplastic train, train chunks

noplastic_train(data['transition_distances'])


