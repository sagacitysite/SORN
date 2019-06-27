import evaluation as ev
from evaluation import *

def clustering(weights_ee, weights_eu):
    print('# Weights: Calculate clusteres')
    # runs / models / train

    # Get parameters and define helpers
    shape = np.shape(weights_ee)
    num_states = len(PARA.c.states)
    alphabet = set("".join(PARA.c.states))
    lookup = dict(zip(alphabet, range(num_states)))
    #num_cluster_neurons = PARA.c.N_u_e  # input neurons per cluster
    #num_other_neurons = (PARA.c.N_e - (num_states*num_cluster_neurons))

    # Prepare arrays to store information
    clusters = np.zeros(ev._helper.flatten((np.shape(weights_ee)[0:3], (num_states,num_states))))
    clusters_all = np.zeros(ev._helper.flatten((np.shape(weights_ee)[0:3], (num_states+1,num_states+1))))
    #sorted_weights = np.zeros(np.shape(weights_ee))

    for g in range(shape[0]):  # runs
        for h in range(shape[1]):  # models
            for k in range(shape[2]):  # train
                # Get weight matrix for current configuration
                tmp_weights_eu = weights_eu[g, h, k, :, :]
                tmp_weights_ee = weights_ee[g, h, k, :, :]  # transpose matrix to get correct form
                
                # Get indices of 'others' (reservoir neurons which are not part of a cluster)
                others = np.sum(tmp_weights_eu, axis=1)
                others[others > 0] = True
                idx_others = np.invert(others.astype(bool))

                # Run through clusters
                for i in range(num_states+1):
                    # Make boolean out of weight matrix
                    # 'true' if neuron has input, 'false' if neuron is 'other' (not part of a cluster)
                    if i < num_states:
                        state_i = lookup[PARA.c.states[i]]  # 'from' state
                        idx_i = tmp_weights_eu[:, state_i].astype(bool)  # Get indices of all neurons in cluster i
                    else:
                        idx_i = idx_others  # Finally, get indices of 'other' neurons
                    
                    # Run through clusters
                    for j in range(num_states+1):
                        if j < num_states:
                            state_j = lookup[PARA.c.states[j]]  # 'to' state
                            idx_j = tmp_weights_eu[:, state_j].astype(bool)   # Get indices of all neurons in cluster j
                        else:
                            idx_j = idx_others  # Finally, get indices of 'other' neurons
                        
                        # Weights between cluster i and j
                        weights_ij = tmp_weights_ee[:, idx_i][idx_j, :]

                        # Mean all weights inside of cluster to obtain coefficient
                        weights_ij_mean = np.mean(weights_ij)

                        # Add weights mean to cluster 'with others' array
                        clusters_all[g, h, k, i, j] = weights_ij_mean
                        # Add weights mean to cluster 'without others' array, only if index does not hit 'others'
                        if i < num_states and j < num_states:
                            clusters[g, h, k, i, j] = weights_ij_mean

                    # Normalize (with others)
                    coeff_sum_all = np.sum(clusters_all[g, h, k, i, :])
                    clusters_all[g, h, k, i, :] = clusters_all[g, h, k, i, :]/coeff_sum_all

                    # Normalize (only without others)
                    if i < num_states:
                        coeff_sum = np.sum(clusters[g, h, k, i, :])
                        clusters[g, h, k, i, :] = clusters[g, h, k, i, :]/coeff_sum

    # Mean and Std over runs (only without others)
    #clusters_mean = np.mean(clusters, axis=0)
    #clusters_sd = np.std(clusters, axis=0)

    # Mean and Std over runs (with and without others)
    #clusters_all_mean = np.mean(clusters_all, axis=0)
    #clusters_all_sd = np.std(clusters_all, axis=0)

    return clusters, clusters_all

"""
@desc: Plots normlaized the mean weight between clusters
       If 'with_others' is True, all excitatory reservoir neurons are considered
       if it is False, inly the cluster neurons are considered
"""
def plot_weight_clusters(clusters, with_others=False):

    for i in range(np.shape(clusters)[0]):  # num_models

        # Prepare values
        curr_cluster = clusters[i]
        num_states = np.size(PARA.c.states)

        # Prepare plot parameters
        num_ticks = np.arange(num_states+1) if with_others else np.arange(num_states)
        ticks = np.append(PARA.c.states, 'Others') if with_others else PARA.c.states

        # Plot
        plt.imshow(curr_cluster, clim=(0,0.5), origin='upper', cmap='copper_r', interpolation='nearest')
        for (k,l),label in np.ndenumerate(curr_cluster):
            col = 'black' if curr_cluster[l,k] < 0.25 else 'white'
            plt.text(k, l, np.around(curr_cluster[l,k], 2), color=col, size=11, ha='center', va='center')
        plt.xlabel('To')
        plt.xticks(num_ticks, ticks)
        plt.ylabel('From')
        plt.yticks(num_ticks, ticks)
        plt.colorbar()
        #plt.title('Model '+str(i+1))
        plt.savefig(PLOTPATH + '/weight_structure_model'+str(i+1)+'.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()

"""
@desc: Plots the weight matrix, where the neurons are sorted by the input clusters
"""
def plot_sorted_weight_matrix(weights_ee, weights_eu):
    # weights: runs / models

    weights_eu

    num_runs = weights_ee.shape[0]
    num_models = weights_ee.shape[1]

    for i in range(num_runs):
        for j in range(num_models):
            ee = weights_ee[i,j]
            
            # Get sort indices of input clusters
            argsrt = np.lexsort(tuple(weights_eu[i,j].T))

            # Sort
            ee = ee[:,argsrt]
            ee = ee[argsrt,:]

            # Store sorted weights
            weights_ee[i,j] = ee

    # Sum over runs
    ee_mean = np.mean(weights_ee, axis=0)

    # Plot for all models
    for i in range(num_models):
        plt.imshow(ee_mean[i], clim=(0,0.05), origin='upper', cmap='copper_r', interpolation='nearest')
        plt.xlabel('To')
        plt.ylabel('From')
        plt.colorbar()
        #plt.title('Model '+str(i+1))
        plt.savefig(PLOTPATH + '/weight_matrix_model'+str(i+1)+'.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()

def errors(clusters):
    # Number of steps, where transition matrix is calculated for
    num_states = np.size(PARA.c.states)

    # Get transition matrices (inital and clusters)
    we = np.moveaxis(clusters, 1, -3)  # Move 'models' axis to the third last one
    it = PARA.c.source.transitions

    # p1 matrix norm of differences
    p1_error = np.sum(np.abs(we - it), axis=(3,4))
    p1_error = np.moveaxis(p1_error, -1, 1)
    p1_error_mean = np.mean(p1_error, axis=0)
    p1_error_sd = np.std(p1_error, axis=0)

    # p2 matrix norm of difference
    p2_error = np.sqrt(np.sum(np.square(we - it), axis=(3,4)))
    p2_error = np.moveaxis(p2_error, -1, 1)
    p2_error_mean = np.mean(p2_error, axis=0)
    p2_error_sd = np.std(p2_error, axis=0)

    return p1_error_mean, p1_error_sd, p2_error_mean, p2_error_sd

def errors_signalnoise(clusters):
    # Number of steps, where transition matrix is calculated for
    num_states = np.size(PARA.c.states)

    # Get transition matrices (inital and clusters)
    we = clusters
    # FIXME This is a fixed transition matrix for 4 states!
    # it_signal = np.array([[0, 1, 0],
    #                      [0, 0, 1],
    #                      [1, 0, 0]])
    # it_noise = np.array([[0.25, 0.25, 0.25],
    #                     [0.25, 0.25, 0.25],
    #                     [0.25, 0.25, 0.25]])
    it_signal = np.array([[0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])
    it_noise = np.array([[0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25]])
    # it_signal = np.array([[0, 1, 0, 0, 0],
    #                       [0, 0, 1, 0, 0],
    #                       [0, 0, 0, 1, 0],
    #                       [0, 0, 0, 0, 1],
    #                       [1, 0, 0, 0, 0]])
    # it_noise = np.array([[0.25, 0.25, 0.25, 0.25, 0.25],
    #                      [0.25, 0.25, 0.25, 0.25, 0.25],
    #                      [0.25, 0.25, 0.25, 0.25, 0.25],
    #                      [0.25, 0.25, 0.25, 0.25, 0.25],
    #                      [0.25, 0.25, 0.25, 0.25, 0.25]])

    # p2 matrix norm regarding signal
    p2_signal = np.sqrt(np.sum(np.square(we - it_signal), axis=(3,4)))
    p2_signal_mean = np.mean(p2_signal, axis=0)
    p2_signal_sd = np.std(p2_signal, axis=0)

    # p2 matrix norm regarding noise
    p2_noise = np.sqrt(np.sum(np.square(we - it_noise), axis=(3,4)))
    p2_noise_mean = np.mean(p2_noise, axis=0)
    p2_noise_sd = np.std(p2_noise, axis=0)

    return p2_signal_mean, p2_signal_sd, p2_noise_mean, p2_noise_sd

def errors_delta(clusters):
    # Number of steps, where transition matrix is calculated for
    num_states = np.size(PARA.c.states)

    # Get transition matrices (inital and clusters)
    we = np.moveaxis(clusters, 1, -3)  # Move 'models' axis to the third last one
    we_next = we
    we_prev = np.roll(we, shift=1, axis=2)
    # Note: Later skip first model difference, since difference between first and last is not in interest

    # p1 matrix norm of differences
    p1_diff = np.sum(np.abs(we - we_prev), axis=(3,4))
    p1_diff = np.moveaxis(p1_diff, -1, 1)[:,1:,:]
    p1_diff_mean = np.mean(p1_diff, axis=0)
    p1_diff_sd = np.std(p1_diff, axis=0)

    # p2 matrix norm of difference
    p2_diff = np.sqrt(np.sum(np.square(we - we_prev), axis=(3,4)))
    p2_diff = np.moveaxis(p2_diff, -1, 1)[:,1:,:]
    p2_diff_mean = np.mean(p2_diff, axis=0)
    p2_diff_sd = np.std(p2_diff, axis=0)

    return p1_diff_mean, p1_diff_sd, p2_diff_mean, p2_diff_sd
