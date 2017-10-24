import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# Path
current = "2017-10-24_14-50-32_hamming_10000"

path = os.getcwd() + "/backup/test_single/" + current
datapath = path + "/data"
plotpath = path + "/plots"

if not os.path.exists(plotpath):
    os.mkdir(plotpath)

# Import parameters
sys.path.insert(0,path+"/michaelis")
import param_mcmc as para

# Load hamming distance data
hamming = np.load(datapath + "/hamming_distances_.npy").astype(int)

# Do integer correction
hamming[hamming < 0] = 0

# Get frequencies for all hamming occurencies
hamming_freqs = np.bincount(hamming)

# Plot and save plot
plt.bar(np.arange(len(hamming_freqs)), hamming_freqs)
plt.title('Training steps: '+str(para.c.steps_plastic))
plt.xlabel('Hamming distance')
plt.ylabel('Frequency')
plt.savefig(plotpath + '/hamming_distances_frequencies.png', dpi=144)
plt.close()

