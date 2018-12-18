from evaluation import *
import glob
import pandas

"""
@desc: Prepare data from files to numpy array
"""
def load(sources):
    print('# Load data')
    
    data = sources
    for d in sources:
        folder = None
        if (d == 'transition_distances'):
            folder = 'transition_norms'
        else:
            folder = d
        # Get folder and store files in arrays
        path = glob.glob(os.path.join(DATAPATH, folder))[0]
        arrays = [np.load(path + '/run' + str(run) + '.npy') for run in range(NUM_RUNS)]
        # Stack arrays to one array
        data[d] = np.stack(arrays, axis=0)
    return data


"""
@desc: Chunk data
"""
def chunk_data(data, axis, moveto):
    # Chunk data
    chunked = np.split(data, NUM_CHUNKS, axis=axis)

    # Move axis and return chunked data
    return np.moveaxis(chunked, 0, moveto)


"""
@desc: Derive statistics from data
"""
def get_statistics():
    activity = []
    hamming_distances = []
    learned_transitions = []
    l1_errors = []
    l2_errors = []
    for i in range(NUM_RUNS):
        print('# Prepare data for run '+str(i+1))

        # Load spontaneous activity for current run
        path = glob.glob(os.path.join(DATAPATH, 'noplastic_test'))[0]
        spont_spikes = np.load(path + '/run' + str(i) + '.npy')

        # Load evoked activity for current run
        path = glob.glob(os.path.join(DATAPATH, 'norm_last_input_spikes'))[0]
        evoked_spikes = np.load(path + '/run' + str(i) + '.npy')

        # Load evoked indices for current run
        path = glob.glob(os.path.join(DATAPATH, 'norm_last_input_index'))[0]
        evoked_indices = np.load(path + '/run' + str(i) + '.npy')

        # Calculate spontaneous activity
        activity.append(spont_activity(spont_spikes))

        # Calculate hamming
        hd, markov_state_indices = classify_spont_activtiy(spont_spikes, evoked_spikes, evoked_indices)
        hamming_distances.append(hd)

        # Calculate transitions matrices
        lt = get_learned_transitions(markov_state_indices)
        learned_transitions.append(lt)

        # Calculate errors
        l1_error, l2_error = get_errors(lt)
        l1_errors.append(l1_error)
        l2_errors.append(l2_error)

        # TODO calculate stationaries
        # Is this still necessary at that place in code?

        # TODO ncomparison
        # Is this still necessary?
    
    # Stack all statistics together
    spont_stats_all = np.stack(activity, axis=0)
    # ...

    # Axes: runs, models, train steps, thresholds, h_ip, #ee-connections, neurons
    return spont_stats_all

"""
@desc: Calculate transition errors
"""
def get_errors(learned_transitions):
    print(np.shape(learned_transitions))
    # Get learned transitions and initally chosen transitions (used to train the network)
    lt = np.moveaxis(learned_transitions, 0, -3)  # Move 'models' axis to the third last one
    it = PARA.c.source.transitions

    # Number of steps, where transition matrix is calculated for
    num_steps = np.size(PARA.c.states)

    print(np.shape(lt))

    # Normalized L1 matrix norm of differences
    l1_error = np.sum(np.abs(lt - it), axis=(6,7))/np.square(num_steps)
    print(np.shape(l1_error))
    l1_error = np.moveaxis(l1_error, 5, 0)
    print(np.shape(l1_error))

    # Normalized L2 matrix norm of difference
    l2_error = np.sqrt(np.sum(np.square(lt - it), axis=(6,7)))/num_steps
    print(np.shape(l2_error))
    l2_error = np.moveaxis(l2_error, 5, 0)
    print(np.shape(l2_error))
    sys.exit()

    # TODO Apply network with 15000 steps (3 chunks) and check if dimensions are correct

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
    for i in range(np.shape(cross_table)[0]):
        # Only normalize if sum of transition is not 0
        if np.sum(cross_table[:, i]) != 0:
            cross_table[:, i] /= np.sum(cross_table[:, i])
    
    return cross_table.T

"""
@desc: Calculate transition matrices
"""
def get_learned_transitions(markov_state_indices):
    # Caculate transitions for all data
    return np.apply_along_axis(calc_crosstable, 6, markov_state_indices)

"""
@desc: Statistic: sponaneous activity for one run
"""
def spont_activity(spont_spikes):
    # Chunk spikes
    spont_spikes = chunk_data(spont_spikes, axis=6, moveto=-3)
    # Mean over neurons and chunks
    spont_activity_mean = np.mean(spont_spikes, axis=(6,7))
    # Move chunk axis to the end of the array (sucht that it replaces steps)
    spont_activity_mean = np.moveaxis(spont_activity_mean, 0, -1)
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
