import evaluation as ev
from evaluation import *

def clustering(weights_ee, weights_eu, with_others=True):
    print('# Weights: Plot clusters')

    clustered_weights = []
    shape = np.shape(weights_ee)
    #weight_threshold = 0.01  #0.0001

    num_states = len(PARA.c.states)
    alphabet = set("".join(PARA.c.states))
    lookup = dict(zip(alphabet, range(num_states)))

    cluster_coefficient = np.zeros(ev._helper.flatten((np.shape(weights_ee)[0:3], (num_states+1,num_states+1))))
    for g in range(shape[0]):  # runs
        for h in range(shape[1]):  # models
            for k in range(shape[2]):  # weight dense
                tmp_weights_eu = weights_eu[g, h, k, :, :]
                tmp_weights_ee = weights_ee[g, h, k, :, :].T  # transpose matrix to get correct form
                
                others = np.sum(tmp_weights_eu, axis=1)
                others[others > 0] = True
                idx_others = np.invert(others.astype(bool))

                for i in range(num_states+1):
                    # Get weights of neurons for input state i and j
                    # and make boolean out of it ('true' if neuron has input, 'false' if not)
                    if i < num_states:
                        state_i = lookup[PARA.c.states[i]]
                        idx_i = tmp_weights_eu[:, state_i].astype(bool)
                    else:
                        idx_i = idx_others
                    
                    for j in range(num_states+1):
                        if j < num_states:
                            state_j = lookup[PARA.c.states[j]]
                            idx_j = tmp_weights_eu[:, state_j].astype(bool)
                        else:
                            idx_i = idx_others
                        
                        # Weights between neurons of state i and state j
                        weights_ij = tmp_weights_ee[idx_i, :][:, idx_j]
                        
                        # Remove weights which are very small
                        #weights_ij[weights_ij < weight_threshold] = 0
                        
                        # Sum all weights to obtain coefficient
                        cluster_coefficient[g, h, k, i, j] = np.mean(weights_ij)
                    
                    # Normalize
                    coeff_sum = np.sum(cluster_coefficient[g, h, k, i, :])
                    cluster_coefficient[g, h, k, i, :] = cluster_coefficient[g, h, k, i, :]/coeff_sum

    cluster_coefficient_mean = np.mean(cluster_coefficient, axis=0)

    for i in range(shape[1]):  # num_models

        # Prepare values
        values = cluster_coefficient_mean[i,0]  # Simply take first 'weight dense'

        # Values without others
        values_without_others = values[:-1,:-1]
        # Renorm
        row_sums = np.sum(values_without_others, axis=1)
        values_without_others = np.array([values_without_others[k,:]/row_sums[k] for k in range(np.shape(values_without_others)[0])])

        # If only the states should be considered and not the connections to the other non-cluster neurons
        if with_others == False:
            values = values_without_others
        
        # Append weight clusters of current model to list
        clustered_weights.append(values)

        # Prepare plot parameters
        num_ticks = np.arange(num_states+1) if with_others else np.arange(num_states)
        ticks = np.append(PARA.c.states, 'Others') if with_others else PARA.c.states

        # Plot
        plt.imshow(values, clim=(0,0.5), origin='upper', cmap='copper_r', interpolation='nearest')
        for (k,l),label in np.ndenumerate(values):
            col = 'black' if values[l,k] < 0.25 else 'white'
            plt.text(k, l, np.around(values[l,k], 2), color=col, size=11, ha='center', va='center')
        plt.xlabel('To')
        plt.xticks(num_ticks, ticks)
        plt.ylabel('From')
        plt.yticks(num_ticks, ticks)
        plt.colorbar()
        #plt.title('Model '+str(i+1))
        plt.savefig(PLOTPATH + '/weight_structure_model'+str(i+1)+'.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()
    
    return np.array(clustered_weights)

def errors(clustered_weights):
    # Number of steps, where transition matrix is calculated for
    num_states = np.size(PARA.c.states)

    # Get initial transition matrix
    it = PARA.c.source.transitions
    we = clustered_weights

    # Normalized L1 matrix norm of differences
    l1_error = np.sum(np.abs(we - it), axis=(1,2))/np.square(num_states)

    # Normalized L2 matrix norm of difference
    l2_error = np.sqrt(np.sum(np.square(we - it), axis=(1,2)))/num_states

    return l1_error, l2_error
