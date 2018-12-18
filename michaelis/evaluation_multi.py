# Import evaluation module
import evaluation as ev
from evaluation import *

"""
Load and prepare data
"""

# transition_distances, activity, estimated_stationaries, ncomparison, hamming_distances

# Load data
sources = ev._configure.sources()
#data = ev._data.load(sources)  # runs, models, train steps, thresholds, h_ip, #ee-connections, test steps / test chunks

stats = ev._data.get_statistics()

print('stats')
print(stats)

sys.exit()

# Indices
hpos_idx = None
if IP == "h_ip_factor":
    hpos_idx = np.where(PARA.c.h_ip_factor == 2.0)[0][0]  # index where h_ip = 2.0
elif IP == "eta_ip":
    hpos_idx = np.where(np.isclose(PARA.c.eta_ip, 0.001))[0][0]  # index where eta_ip = 0.001
elif IP == "h_ip_range":
    hpos_idx = np.where(np.isclose(PARA.c.h_ip_range, 0.01))[0][0]  # index where h_ip_range = 0.01
    
mxthresh_idx = np.shape(data['transition_distances'])[3]-1  # index where hamming threshold is max
dens_idx = np.where(PARA.c.connections_density == 0.1)[0][0]  # index where connections_density = 0.1
train_idx = len(PARA.c.steps_plastic)-1  # index where training steps are max

# Prepare some data sets
normed_stationaries = ev._helper.norm_stationaries(data['estimated_stationaries'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:])
stationary_distances = ev._helper.calc_stationary_distances(normed_stationaries)

#hamming_means = ev.helper.prepare_hamming(data['hamming_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:])
#ncomparison = data['ncomparison'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:]

"""
Training step plots
"""

# Plot transition performance for different training steps
if np.shape(data['transition_distances'])[2] > 1:
    training.plot_steps(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:], suffix="transition", ytext="Transition error")

# Plot stationary performance for different training steps
if np.shape(data['estimated_stationaries'])[2] > 1:
    training.plot_steps(stationary_distances, suffix = "stationary", ytext = "Stationary error")
    
# Plot hamming mean for different training steps
#if np.shape(data['hamming_distances'])[2] > 1:
#    training.plot_steps(hamming_means, suffix="hamming", ytext="Relative mean hamming distance", leg_loc=4)

"""
Test trace chunk plots
"""

if len(PARA.c[IP]) > 1:
    y_max = np.max(np.array([np.max(data['transition_distances'][:, :, train_idx, mxthresh_idx, 0, dens_idx, :]),
                     np.max(data['transition_distances'][:, :, train_idx, mxthresh_idx, len(para.c[ip])-1, dens_idx, :])]))
    test_trace.plot(data['transition_distances'][:,:,:,mxthresh_idx,0,dens_idx,:],
                    suffix="distances_smallhip", ylabel="Transition error", ymax=0.5*y_max)
    test_trace.plot(data['transition_distances'][:, :, :, mxthresh_idx, len(PARA.c[ip])-1, dens_idx, :],
                    suffix="distances_hugehip", ylabel="Transition error", ymax=0.5*y_max)

if len(PARA.c.connections_density) > 1:
    y_max = np.max(np.array([np.max(data['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, 0, :]),
                     np.max(data['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, len(PARA.c.connections_density)-1, :])]))

    for i in range(len(PARA.c.connections_density)):
        test_trace.plot(data['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, i, :],
                        suffix="distances_connectivity_train-max_"+str(PARA.c.connections_density[i]), ylabel="Transition error", ymax=0.5*y_max)
    #for i in range(len(para.c.connections_density)):
    #    test_trace.plot(data['transition_distances'][:, :, 0, mxthresh_idx, hpos_idx, i, :],
    #                    suffix="distances_connectivity_train-min_" + str(i), ylabel="Transition error",
    #                    title="connectivity: " + str(para.c.connections_density[i]), ymax=0.5 * y_max)
    # test_trace.plot(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,0,:],
    #                 suffix="distances_sparse_connectivity", ylabel="Transition error", title="sparse connectivity", ymax=0.5*y_max)
    # test_trace.plot(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:],
    #                 suffix="distances_medium_connectivity", ylabel="Transition error", title="medium connectivity", ymax=0.5*y_max)
    # test_trace.plot(data['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, len(para.c.connections_density)-1, :],
    #                 suffix="distances_dense_connectivity", ylabel="Transition error", title="dense connectivity", ymax=0.5*y_max)

test_trace.plot(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:],
                suffix="distances", ylabel="Transition error")#, ymax=0.3)
test_trace.plot(stationary_distances[:,:,:,:], suffix="stationary", ylabel="Stationary error")

"""
Hamming Distancs evaluation
"""

#if np.shape(data['hamming_distances'])[3] > 1:
#    hamming.plot_histogram(data['hamming_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:])

#if np.shape(data['transition_distances'])[2] > 1:
#    training.plot_thresholds(data['transition_distances'][:,:,:,:,hpos_idx,dens_idx,:])

"""
Activity/NComparison
"""

#activity_distance_correlation_plot(distances, activity)
#performance_correlation.plot_ncomparison(data['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:], ncomparison[:,:,:,np.shape(ncomparison)[3]-1])

"""
Activity/H_IP
"""

if ip == "h_ip_factor":
    ip.plot(data['transition_distances'][:,:,train_idx,mxthresh_idx,:,dens_idx,:])
    ip.plot_activity(data['activity'][:,:,train_idx,mxthresh_idx,:,dens_idx,:])
    activity.plot_trace(data['activity'][0,:,train_idx,mxthresh_idx,hpos_idx,dens_idx,:])
    
if ip == "h_ip_range":
    ip.plot(data['transition_distances'][:,:,train_idx,mxthresh_idx,:,dens_idx,:], legend_labels="$\sigma^{IP}$")

"""
Variance, Entropy, Gini / Lorenz
"""

performance_correlation.plot_inequality(data['transition_distances'][:,:,:,mxthresh_idx,:,dens_idx,:], hpos_idx)

#lorenz_plot(normed_stationaries)

"""
Connectivity
"""

if len(PARA.c.connections_density) > 1:
    connectivity.plot_performance(data['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, :, :])
    
"""
Activity structur
"""

# runs, models, ..., neurons, steps
#spikes.plot_train_histograms(data['norm_last_input_spikes'][0,:,train_idx,mxthresh_idx,hpos_idx,dens_idx,:,:])  # get first run only

"""
Weight Strength
"""

max_train = np.shape(data['weights_ee'])[2]-1
weights_ee = data['weights_ee'][:,:,max_train,mxthresh_idx,hpos_idx,:]
weights_eu = data['weights_eu'][:,:,max_train,mxthresh_idx,hpos_idx,:]
# now: runs / models / weight dense

weights.clustering(weights_ee, weights_eu)
