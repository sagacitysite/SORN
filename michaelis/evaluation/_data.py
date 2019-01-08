from evaluation import *
import evaluation as ev
import glob
import pandas

"""
@desc: Derive statistics from data
"""
def get_statistics():
    # Initalize stats dictionary
    stats = {
        'activity': [],
        'hamming_distances': [],
        'transition_matrices': [],
        'stationary_distributions': [],
        'l1_errors': [],
        'l2_errors': [],
        'weights_ee': [],
        'weights_eu': [] }
    
    # If statistics are already cached, just get them from cache file
    cache_path = DATAPATH + '/statistics_cache.npz'
    if os.path.exists(cache_path):
        print('# Load statistics')
        return np.load(cache_path)    

    # If statistics are not cached, run the procedure
    sys.stdout.write('# Prepare statistics ')
    sys.stdout.flush()

    for i in range(NUM_RUNS):
        sys.stdout.write('.')
        sys.stdout.flush()

        """
        Load data from selected backup folder
        """
        # Load spontaneous activity for current run
        path = glob.glob(os.path.join(DATAPATH, 'noplastic_test'))[0]
        spont_spikes = np.load(path + '/run' + str(i) + '.npy')

        # Load evoked activity for current run
        path = glob.glob(os.path.join(DATAPATH, 'norm_last_input_spikes'))[0]
        evoked_spikes = np.load(path + '/run' + str(i) + '.npy')

        # Load evoked indices for current run
        path = glob.glob(os.path.join(DATAPATH, 'norm_last_input_index'))[0]
        evoked_indices = np.load(path + '/run' + str(i) + '.npy')

        # Load weights EE
        path = glob.glob(os.path.join(DATAPATH, 'weights_ee'))[0]
        stats['weights_ee'].append(np.load(path + '/run' + str(i) + '.npy'))

        # Load weights EU
        path = glob.glob(os.path.join(DATAPATH, 'weights_eu'))[0]
        stats['weights_eu'].append(np.load(path + '/run' + str(i) + '.npy'))

        """
        Calculate statistcs, based on the loaded data
        """
        # Calculate spontaneous activity
        stats['activity'].append(spont_activity(spont_spikes))

        # Calculate hamming
        hd, markov_state_indices = classify_spont_activtiy(spont_spikes, evoked_spikes, evoked_indices)
        stats['hamming_distances'].append(hd)

        # Calculate transitions matrices and stationary distributions
        lt, sd = get_learned_transitions_and_stationaries(markov_state_indices)
        stats['transition_matrices'].append(lt)
        stats['stationary_distributions'].append(sd)

        # Calculate errors
        l1_error, l2_error = get_errors(lt)
        stats['l1_errors'].append(l1_error)
        stats['l2_errors'].append(l2_error)

        # TODO ncomparison
        # Is this still necessary?

    print(' .')

    # Stack all statistics together
    for key in stats:
        stats[key] = np.stack(stats[key], axis=0)

    # Cache results
    np.savez_compressed(cache_path, **stats)

    # Axes: runs, models, train steps, thresholds, h_ip, #ee-connections, neurons
    return stats

"""
@desc: Chunk data
"""
def chunk_data(data, axis, moveto):
    # Chunk data (note: be carful, NUM_CHUNKS is calculated initially, depending on overall available data)
    chunked = np.split(data, NUM_CHUNKS, axis=axis)

    # Move axis and return chunked data
    return np.moveaxis(chunked, 0, moveto)

"""
@desc: Calculate transition errors
"""
def get_errors(transition_matrices):
    # Get learned transitions and initally chosen transitions (used to train the network)
    lt = np.moveaxis(transition_matrices, 0, -3)  # Move 'models' axis to the third last one
    it = PARA.c.source.transitions

    # Number of steps, where transition matrix is calculated for
    num_states = np.size(PARA.c.states)

    # Normalized L1 matrix norm of differences
    l1_error = np.sum(np.abs(lt - it), axis=(6,7))/np.square(num_states)
    l1_error = np.moveaxis(l1_error, 5, 0)

    # Normalized L2 matrix norm of difference
    l2_error = np.sqrt(np.sum(np.square(lt - it), axis=(6,7)))/num_states
    l2_error = np.moveaxis(l2_error, 5, 0)

    return l1_error, l2_error

"""
@desc: Helper function, uses pandas crosstab function to calculate cross table of indices
"""
def calc_crosstable(ind):
    # Arguments for crosstab function: from = ind[:-1], to = ind[1:]
    # Transform pandas format to numpy array
    cross_table = np.array(pandas.crosstab(ind[:-1], ind[1:])).astype(float)

    # If some states are missing in the cross table, throw an error
    if np.shape(cross_table)[0] < np.size(PARA.c.states):
        raise Exception(
                'Some states did not occure. Probably the number of test steps needs to be increased')

    # Normalize cross table
    row_sums = np.sum(cross_table, axis=1)
    for i in range(np.shape(cross_table)[0]):
        # Only normalize if sum of transition is not 0
        if row_sums[i] != 0:
            cross_table[i,:] /= row_sums[i]
    
    return cross_table

"""
@desc: Helper function to prepare data for calling stationary_distribution() function
       A reshaped matrix is given, which is reshaped again in normal squared matrix form,
       before it is applied to the stationary_distribution() function
"""
def calc_stationary_distribution(transition_matrix_reshaped):
    # Reshape transition matrix in squared form
    num_states = np.size(PARA.c.states)
    transition_matrix = np.reshape(transition_matrix_reshaped, (num_states, num_states))

    # Calculate stationary distribution
    stationary = ev._helper.stationary_distribution(transition_matrix)
    return stationary

"""
@desc: Calculate transition matrices
"""
def get_learned_transitions_and_stationaries(markov_state_indices):
    """
    Calculate transition matices
    """
    # Caculate transitions for all data
    transition_matrices = np.apply_along_axis(calc_crosstable, 6, markov_state_indices)

    """
    Calculate stationary distributions from transitions matrices
    """
    # Prepare reshape dimensions
    shp = list(np.shape(transition_matrices))  # Transform shape to list
    shp = shp[:len(shp)-2]  # Remove last two dimensions
    shp.append(np.square(np.size(PARA.c.states)))  # Add squared dimension
    new_shape = tuple(shp)  # Transform to tuple

    # Reshape matrix to be ready to call apply_along_axis
    transition_matrices_reshaped = np.reshape(transition_matrices, new_shape)
    
    # Call helper function, which calculates stationary distributions for all data
    stationary_distributions = np.apply_along_axis(calc_stationary_distribution, 6, transition_matrices_reshaped)

    return transition_matrices, stationary_distributions

"""
@desc: Statistic: sponaneous activity for one run
"""
def spont_activity(spont_spikes):
    # Chunk spikes
    spont_spikes = chunk_data(spont_spikes, axis=6, moveto=-3)
    # Mean over neurons and chunks
    spont_activity_mean = np.mean(spont_spikes, axis=(6,7))

    # Axes: models, train steps, thresholds, h_ip, #ee-connections, chunks
    return spont_activity_mean

"""
@desc: Calculate hamming distances and find markov states in spontaneous activity
"""
def classify_spont_activtiy(spont_spikes, evoked_spikes, evoked_indices):
    # Number of spontaneous trials = noplastic testing trials, including silent ones
    N_spont = np.shape(spont_spikes)[6]

    # Define empty array for hamming distances
    smallest_hamming_distances = []
    markov_state_indices = []

    # Find for each spontaneous state (= noplastic test) state the evoked state (= noplastic train) with the
    # smallest hamming distance and store the corresponding index
    for i in range(N_spont):
        # One spontaneous state (= noplastic test) is subtracted from all input states from noplastic training phase (broadcasting is used)
        # it is searched for the minimal value, which results in the most similar evoked index
        h = np.sum(np.abs(evoked_spikes - spont_spikes[:,:,:,:,:,:,i,np.newaxis]), axis=5)
        most_similar_index = np.argmin(h, axis=5)

        # Hamming distance between most_similar state and current spontaneous state
        shp = [np.arange(x) for x in h.shape]
        hd = h[shp[0], shp[1], shp[2], shp[3], shp[4], np.squeeze(most_similar_index)]
        hd = np.reshape(hd, tuple(np.asarray(h.shape)[:-1]))
        
        smallest_hamming_distances.append(hd)

        # Classify markov states
        shp = [np.arange(x) for x in evoked_indices.shape]
        msi = evoked_indices[shp[0], shp[1], shp[2], shp[3], shp[4], np.squeeze(most_similar_index)]
        msi = np.reshape(msi, tuple(np.asarray(h.shape)[:-1]))

        markov_state_indices.append(msi)

    shd = chunk_data(np.stack(smallest_hamming_distances, axis=5), axis=5, moveto=-2)
    msi = chunk_data(np.stack(markov_state_indices, axis=5), axis=5, moveto=-2)

    return shd, msi
