# Import evaluation module
import evaluation as ev
from evaluation import *

"""
Load and prepare data
"""

# Load data
stats = ev._data.get_statistics()
# DICTIONARY: activity, transitions_matrices, stationary_distributions, hamming_distances, p1_errors, p2_errors, weights_ee, weights_eu
# DIMENSIONS: runs, models, train steps, thresholds, h_ip, #ee-connections, test chunks / test steps

# Rename old nomenclatur
if 'l1_errors' in stats:
    stats['p1_errors'] = stats.pop('l1_errors')
if 'l2_errors' in stats:
    stats['p2_errors'] = stats.pop('l2_errors')

# Indices
hpos_idx = None
if IP == "h_ip_factor":
    hpos_idx = np.where(PARA.c.h_ip_factor == 2.0)[0][0]  # index where h_ip = 2.0
elif IP == "eta_ip":
    hpos_idx = np.where(np.isclose(PARA.c.eta_ip, 0.001))[0][0]  # index where eta_ip = 0.001
elif IP == "h_ip_range":
    hpos_idx = np.where(np.isclose(PARA.c.h_ip_range, 0.01))[0][0]  # index where h_ip_range = 0.01
    
mxthresh_idx = np.shape(stats['p2_errors'])[3]-1  # index where hamming threshold is max
dens_idx = np.where(PARA.c.connections_density == 0.1)[0][0]  # index where connections_density = 0.1
train_idx = len(PARA.c.steps_plastic)-1  # index where training steps are max

# Prepare some data sets
normed_stationaries = ev._helper.norm_stationaries(stats['stationary_distributions'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:])
stationary_distances = ev._helper.calc_stationary_distances(normed_stationaries)

#hamming_means = ev.helper.prepare_hamming(stats['hamming_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:])
#ncomparison = stats['ncomparison'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:]

"""
Sequence counting: How often does the sequence ABCD occur?
"""

#activity.count_sequences(stats['markov_state_indices'][:,:,train_idx,mxthresh_idx,hpos_idx,dens_idx,:])

"""
Firing Thresholds
"""

# print('# Firing thresholds')

# # Set path
# import glob
# path = glob.glob(os.path.join(DATAPATH, 'firing_thresholds'))[0]

# # Get an mean thresholds
# firing_thresholds_mean = []
# firing_thresholds_sd = []
# for i in range(NUM_RUNS): 
#     ft = np.load(path + '/run'+str(i)+'.npy')
#     ft_mean = np.mean(ft, axis=(5,6))
#     ft_sd = np.std(ft, axis=(5,6))
#     firing_thresholds_mean.append(ft_mean)
#     firing_thresholds_sd.append(ft_sd)

# # Stack all runs together
# firing_thresholds_mean = np.stack(firing_thresholds_mean, axis=0)
# firing_thresholds_sd = np.stack(firing_thresholds_sd, axis=0)

# # Prepare data
# num_train_steps = np.shape(ft_mean)[1]
# ft_mean_mean = np.mean(firing_thresholds_mean, axis=0)
# ft_mean_sd = np.std(firing_thresholds_mean, axis=0)
# ft_sd_mean = np.mean(firing_thresholds_sd, axis=0)
# ft_sd_sd = np.std(firing_thresholds_sd, axis=0)
# transition_entropy = np.array([ev._helper.transition_entropy(t) for t in PARA.c.source.transitions])

# # Mean thresholds entropy max train
# plt.errorbar(transition_entropy, ft_mean_mean[:,-1], yerr=ft_mean_sd[:,-1], fmt='o')
# plt.ylim(ymin=0)
# plt.legend(prop={'size': LEGEND_SIZE})
# plt.xlabel('entropy rate', color=FIG_COLOR)
# plt.ylabel('firing threshold', color=FIG_COLOR)
# plt.savefig(PLOTPATH + '/correlation_thresholds_entropy.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
# plt.close()

# # Mean thresholds entropy train
# if num_train_steps > 1:
#     for i in range(num_train_steps):
#         legend = 'static reservoir' if ((i == 0) and (PARA.c.steps_plastic[0] == 0)) else str(PARA.c.steps_plastic[i]) + ' training steps'
#         plt.errorbar(transition_entropy, ft_mean_mean[:,i], yerr=ft_mean_sd[:,i], fmt='o', label=legend)

#     plt.ylim(ymin=0)
#     plt.legend(prop={'size': LEGEND_SIZE})
#     plt.xlabel('entropy rate', color=FIG_COLOR)
#     plt.ylabel('firing threshold', color=FIG_COLOR)
#     plt.savefig(PLOTPATH + '/correlation_thresholds_entropy_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
#     plt.close()

# # SD thresholds entropy max train
# plt.errorbar(transition_entropy, ft_sd_mean[:,-1], yerr=ft_sd_sd[:,-1], fmt='o')
# plt.ylim(ymin=0)
# plt.legend(prop={'size': LEGEND_SIZE})
# plt.xlabel('entropy rate std', color=FIG_COLOR)
# plt.ylabel('firing threshold', color=FIG_COLOR)
# plt.savefig(PLOTPATH + '/correlation_thresholds_entropy_sd.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
# plt.close()

# # SD thresholds entropy train
# if num_train_steps > 1:
#     for i in range(num_train_steps):
#         legend = 'static reservoir' if ((i == 0) and (PARA.c.steps_plastic[0] == 0)) else str(PARA.c.steps_plastic[i]) + ' training steps'
#         plt.errorbar(transition_entropy, ft_sd_mean[:,i], yerr=ft_sd_sd[:,i], fmt='o', label=legend)

#     plt.ylim(ymin=0)
#     plt.legend(prop={'size': LEGEND_SIZE})
#     plt.xlabel('entropy rate std', color=FIG_COLOR)
#     plt.ylabel('firing threshold', color=FIG_COLOR)
#     plt.savefig(PLOTPATH + '/correlation_thresholds_entropy_sd_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
#     plt.close()


"""
Training step plots
"""

# Plot transition performance for different training steps
if np.shape(stats['p2_errors'])[2] > 1:
    training.plot_steps(stats['p2_errors'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:], suffix="transition", ytext="Transition error")

# Plot stationary performance for different training steps
if np.shape(stats['stationary_distributions'])[2] > 1:
    training.plot_steps(stationary_distances, suffix = "stationary", ytext = "Stationary error")
    
# Plot hamming mean for different training steps
#if np.shape(stats['hamming_distances'])[2] > 1:
#    training.plot_steps(hamming_means, suffix="hamming", ytext="Relative mean hamming distance", leg_loc=4)

"""
Test trace chunk plots
"""

if len(PARA.c[IP]) > 1:
    y_max = np.max(np.array([np.max(stats['p2_errors'][:, :, train_idx, mxthresh_idx, 0, dens_idx, :]),
                     np.max(stats['p2_errors'][:, :, train_idx, mxthresh_idx, len(para.c[ip])-1, dens_idx, :])]))
    test_trace.plot(stats['p2_errors'][:,:,:,mxthresh_idx,0,dens_idx,:],
                    suffix="distances_smallhip", ylabel="Transition error", ymax=0.5*y_max)
    test_trace.plot(stats['p2_errors'][:, :, :, mxthresh_idx, len(PARA.c[ip])-1, dens_idx, :],
                    suffix="distances_hugehip", ylabel="Transition error", ymax=0.5*y_max)

if len(PARA.c.connections_density) > 1:
    y_max = np.max(np.array([np.max(stats['p2_errors'][:, :, :, mxthresh_idx, hpos_idx, 0, :]),
                     np.max(stats['transition_distances'][:, :, :, mxthresh_idx, hpos_idx, len(PARA.c.connections_density)-1, :])]))

    for i in range(len(PARA.c.connections_density)):
        test_trace.plot(stats['p2_errors'][:, :, :, mxthresh_idx, hpos_idx, i, :],
                        suffix="distances_connectivity_train-max_"+str(PARA.c.connections_density[i]), ylabel="Transition error", ymax=0.5*y_max)
    #for i in range(len(para.c.connections_density)):
    #    test_trace.plot(stats['p2_errors'][:, :, 0, mxthresh_idx, hpos_idx, i, :],
    #                    suffix="distances_connectivity_train-min_" + str(i), ylabel="Transition error",
    #                    title="connectivity: " + str(para.c.connections_density[i]), ymax=0.5 * y_max)
    # test_trace.plot(stats['p2_errors'][:,:,:,mxthresh_idx,hpos_idx,0,:],
    #                 suffix="distances_sparse_connectivity", ylabel="Transition error", title="sparse connectivity", ymax=0.5*y_max)
    # test_trace.plot(stats['p2_errors'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:],
    #                 suffix="distances_medium_connectivity", ylabel="Transition error", title="medium connectivity", ymax=0.5*y_max)
    # test_trace.plot(stats['p2_errors'][:, :, :, mxthresh_idx, hpos_idx, len(para.c.connections_density)-1, :],
    #                 suffix="distances_dense_connectivity", ylabel="Transition error", title="dense connectivity", ymax=0.5*y_max)

test_trace.plot(stats['p2_errors'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:],
                suffix="distances", ylabel="Transition error")#, ymax=0.3)
test_trace.plot(stationary_distances[:,:,:,:], suffix="stationary", ylabel="Stationary error")

"""
Hamming Distancs evaluation
"""

#if np.shape(data['hamming_distances'])[3] > 1:
#    hamming.plot_histogram(stats['hamming_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:])

#if np.shape(data['p2_errors'])[2] > 1:
#    training.plot_thresholds(stats['p2_errors'][:,:,:,:,hpos_idx,dens_idx,:])

"""
Activity/NComparison
"""

#activity_distance_correlation_plot(distances, activity)
#performance_correlation.plot_ncomparison(stats['transition_distances'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,:], ncomparison[:,:,:,np.shape(ncomparison)[3]-1])

"""
Activity/H_IP
"""

if ip == "h_ip_factor":
    ip.plot(stats['p2_errors'][:,:,train_idx,mxthresh_idx,:,dens_idx,:])
    ip.plot_activity(stats['activity'][:,:,train_idx,mxthresh_idx,:,dens_idx,:])
    activity.plot_trace(stats['activity'][0,:,train_idx,mxthresh_idx,hpos_idx,dens_idx,:])
    
if ip == "h_ip_range":
    ip.plot(stats['p2_errors'][:,:,train_idx,mxthresh_idx,:,dens_idx,:], legend_labels="$\sigma^{IP}$")

"""
Variance, Entropy, Gini / Lorenz
"""

performance_correlation.plot_inequality(stats['p2_errors'][:,:,:,mxthresh_idx,:,dens_idx,:], hpos_idx)

#lorenz_plot(normed_stationaries)

"""
Density of connections
"""

if len(PARA.c.connections_density) > 1:
    connectivity.plot_performance(stats['p2_errors'][:, :, :, mxthresh_idx, hpos_idx, :, :])
    
"""
Activity structur
"""

# runs, models, ..., neurons, steps
#spikes.plot_train_histograms(stats['norm_last_input_spikes'][0,:,train_idx,mxthresh_idx,hpos_idx,dens_idx,:,:])  # get first run only

"""
Weight Analysis
"""

max_train = np.shape(stats['weights_ee'])[2]-1
weights_ee = stats['weights_ee'][:,:,:,mxthresh_idx,hpos_idx,dens_idx]
weights_eu = stats['weights_eu'][:,:,:,mxthresh_idx,hpos_idx,dens_idx]
# now: runs / models / train

# Get clusters from weight matrix
clusters, clusters_all = weights.clustering(weights_ee, weights_eu)
# now: runs, models, train, transitions (incl. others)

# Calculate error
p1_mean_weights, p1_sd_weights, p2_mean_weights, p2_sd_weights = weights.errors(clusters)

# Plot whole weight matrix, sorted by input clusters
weights.plot_sorted_weight_matrix(weights_ee[:,:,-1], weights_eu[:,:,-1])

# Plot error
performance.barplot_weights_p2(p2_mean_weights[:,max_train], p2_sd_weights[:,max_train])

# Plot errors in relation to entropy rate
performance_correlation.plot_entropy_weights(p2_mean_weights, p2_sd_weights)

# Calculate signal/noise error for weight clusters
p2_mean_weights_signal, p2_sd_weights_signal, p2_mean_weights_noise, p2_sd_weights_noise = weights.errors_signalnoise(clusters)

# Plot errors in relation to signal/noise
#performance_correlation.plot_entropy_signalnoise_weights(
#    p2_mean_weights_signal[:,max_train], p2_sd_weights_signal[:,max_train],
#    p2_mean_weights_noise[:,max_train], p2_sd_weights_noise[:,max_train])

performance_correlation.plot_entropy_signalnoise_weights(
    p2_mean_weights_signal, p2_sd_weights_signal,
    p2_mean_weights_noise, p2_sd_weights_noise)

# Plot weight clusters
weights.plot_weight_clusters(np.mean(clusters, axis=0)[:,max_train])

# Calculate delta error
p1_mean_weights_delta, p1_sd_weights_delta, p2_mean_weights_delta, p2_sd_weights_delta = weights.errors_delta(clusters)

# Plot delta errors
performance.barplot_weights_p2_delta(p2_mean_weights_delta[:,max_train], p2_sd_weights_delta[:,max_train])

"""
Performance spontaneous activity
"""

# Takes last chunk
p2_err = np.mean(stats['p2_errors'][:,:,max_train,mxthresh_idx,hpos_idx,dens_idx,-1], axis=0)
p1_err = np.mean(stats['p1_errors'][:,:,max_train,mxthresh_idx,hpos_idx,dens_idx,-1], axis=0)

# Plot transition errors
performance.barplot_spont(stats['p2_errors'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,-1], filename='error_p2_spont', ylabel='p2 error')

# Calculate signal/noise error for spont transitions
learned_transitions = stats['transition_matrices'][:,:,:,mxthresh_idx,hpos_idx,dens_idx,-1]
p2_mean_spont_signal, p2_sd_spont_signal, p2_mean_spont_noise, p2_sd_spont_noise = performance.errors_signalnoise(learned_transitions)

# Plot
#performance_correlation.plot_entropy_signalnoise_spont(
#    p2_mean_spont_signal[:,max_train], p2_sd_spont_signal[:,max_train],
#    p2_mean_spont_noise[:,max_train], p2_sd_spont_noise[:,max_train])

performance_correlation.plot_entropy_signalnoise_spont(
    p2_mean_spont_signal, p2_sd_spont_signal,
    p2_mean_spont_noise, p2_sd_spont_noise)

"""
Store p1 and p2 errors regarding spontaneous activity and weights
"""

#print(p1_err)
#print(p2_err)
#print(p1_errors_weights)
#print(p2_errors_weights)

"""
Entropy
"""

entropy.initial_weights(clusters[:,:,max_train])

entropy.initial_spont(learned_transitions[:,:,max_train])

"""
Initial and learned transitions matrix
"""

learned_transitions = stats['transition_matrices'][:,:,max_train,mxthresh_idx,hpos_idx,dens_idx,-1]
# now: runs / models

transition_matrix.plot_learned(learned_transitions)
transition_matrix.plot_initial()

"""
Manual stuff
"""


#performance.manual_plots()

performance_correlation.plot_human_experiment()
