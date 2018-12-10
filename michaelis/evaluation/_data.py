from evaluation import *
import glob

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
@desc: Derive statistics from data
"""
def get_statistics():
    activity = []
    hamming_distances = []
    learned_transitions = []
    for i in range(NUM_RUNS):
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

        sys.exit()

        # Calculate hamming
        hd, markov_state_indices = find_states(spont_spikes, evoked_spikes, evoked_indices)
        hamming_distances.append(hd)

        # TODO calculate transitions matrices
        #learned_transitions.append(get_learned_transitions(markov_state_indices))
        
        # TODO calculate transition errors
        # TODO calculate stationaries

        sys.exit()

        # ...
    
    # Stack all statistics together
    spont_stats_all = np.stack(activity, axis=0)
    # ...

    #print(np.shape(spont_stats_all))
    # Axes: runs, models, train steps, thresholds, h_ip, #ee-connections, neurons, chunks

    return spont_stats_all


"""
@desc: Calculate transition matrices
"""
def get_learned_transitions(markov_state_indices):

    # TODO

    maxindex = self.c.maxindex
    transitions = np.zeros((maxindex + 1, maxindex + 1))

    # From all steps count transitions
    for (i_from, i_to) in zip(input[:-1], input[1:]):
        transitions[int(i_to), int(i_from)] += 1

    # Normalize transitions
    for i in range(shape(transitions)[0]):
        # Only normalize if sum of transition is not 0
        if sum(transitions[:, i]) != 0:
            transitions[:, i] /= sum(transitions[:, i])

    return(transitions)

"""
@desc: Statistic: sponaneous activity for one run
"""
def spont_activity(spont_spikes):
    chunk_size = PARA.c.stats.transition_step_size
    spont_spikes_steps = PARA.c.steps_noplastic_test
    num_transition_steps = int(round((spont_spikes_steps / chunk_size) - 0.5))

    # Chunk the activity
    spont_activity_chunks = [spont_spikes[:,:,:,:,:,:,i*chunk_size:(i+1)*chunk_size] for i in range(num_transition_steps)]
    # Mean over neurons and chunks
    spont_activity_chunks_mean = np.mean(spont_activity_chunks, axis=(6,7))
    # Move chunk axis to the end of the array (sucht that it replaces steps)
    spont_activity_chunks_mean = np.moveaxis(spont_activity_chunks_mean, 0, -1)
    # Axes: models, train steps, thresholds, h_ip, #ee-connections, chunks

    return spont_activity_chunks_mean

"""
@desc: Calculate hamming distances and find markov states in spontaneous activity
"""
def find_states(spont_spikes, evoked_spikes, evoked_indices):
    # Number of spontaneous trials = noplastic testing trials, including silent ones
    N_spont = np.shape(spont_spikes)[6]

    # Define empty array for hamming distances
    hamming_distances = np.zeros(N_spont)

    # Find for each spontaneous state (= noplastic test) state the evoked state (= noplastic train) with the
    # smallest hamming distance and store the corresponding index
    markov_state_indices = np.zeros(N_spont)
    for i in range(N_spont):
        # One spontaneous state (= noplastic test) is subtracted from all input states from noplastic training phase (broadcasting is used)
        # it is searched for the minimal value, which results in the most similar evoked index
        most_similar_index = np.argmin(np.sum(np.abs(evoked_spikes - spont_spikes[:,:,:,:,:,:,i,np.newaxis]), axis=5))

        # Hamming distance between most_similar state and current spontaneous state i
        hamming_distances[i] = np.sum(np.abs(evoked_spikes[:,:,:,:,:,:,most_similar_index] - spont_spikes[:,:,:,:,:,:,i]))

        # If current noplastic testing state is NOT silent
        if np.sum(spont_spikes[:,:,:,:,:,:,i]) > 0:
            if not PARA.c.stats.has_key('hamming_threshold'):
                # If no threshold is given, just assign states
                markov_state_indices[i] = evoked_indices[:,:,:,:,:,most_similar_index]
            else:
                # If threshold was met, apply state otherwise apply silent
                markov_state_indices[i] = evoked_indices[:,:,:,:,:,most_similar_index] if hamming_distances[i] < PARA.c.stats.hamming_threshold else -1

        # If current state IS silent
        else:
            markov_state_indices[i] = -1

    return hamming_distances, markov_state_indices

