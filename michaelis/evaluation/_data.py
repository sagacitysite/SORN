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
        #activity.append(spont_activity(evoked_spikes, spont_spikes))

        # Calculate 
        calc_hamming_distances(spont_spikes, evoked_spikes, evoked_indices)

        # ...
    
    # Stack all statistics together
    spont_stats_all = np.stack(activity, axis=0)
    # ...

    #print(np.shape(spont_stats_all))
    # Axes: runs, models, train steps, thresholds, h_ip, #ee-connections, neurons, chunks

    return spont_stats_all

"""
@desc: Statistic: sponaneous activity for one run
"""
def spont_activity(spont_spikes):
    transition_step_size = PARA.c.stats.transition_step_size
    spont_spikes_steps = PARA.c.steps_noplastic_test
    num_transition_steps = int(round((spont_spikes_steps / transition_step_size) - 0.5))

    # Create chunks
    spont_activity_chunks = [spont_spikes[:,:,:,:,:,:,i*transition_step_size:(i+1)*transition_step_size] for i in range(num_transition_steps)]
    # Mean over chunks
    spont_activity_chunks_mean = np.mean(spont_activity_chunks, axis=7)
    # Move chunk axis to the end of the array (sucht that it replaces steps)
    spont_activity_chunks_mean = np.moveaxis(spont_activity_chunks_mean, 0, -1)
    # Axes: models, train steps, thresholds, h_ip, #ee-connections, neurons, chunks

    return spont_activity_chunks_mean

def calc_hamming_distances(spont_spikes, evoked_spikes, evoked_indices):
    # Number of spontaneous trials = noplastic testing trials, including silent ones
    N_spont = np.shape(spont_spikes)[1]

    # Define empty array for hamming distances
    hamming_distances = np.empty(N_spont)

    # Find for each spontaneous state (= noplastic test) state the evoked state (= noplastic train) with the
    # smallest hamming distance and store the corresponding index
    similar_input = np.zeros(N_spont)
    for i in xrange(N_spont):
        # One spontaneous state (= noplastic test) is subtracted from all input states from noplastic training phase (broadcasting is used)
        most_similar = np.argmin(np.sum(np.abs(evoked_spikes.T - spont_spikes[:, i]), axis=1))

        hamming_distances[i] = np.sum(np.abs(evoked_spikes[:, most_similar] - spont_spikes[:, i]))

        # If current noplastic testing state is NOT silent
        if np.sum(spont_spikes[:,i])>0:
            if not PARA.c.stats.has_key('hamming_threshold'):
                # If no threshold is given, just assign states
                similar_input[i] = evoked_indices[most_similar]
            else:
                # If threshold was met, apply state otherwise apply silent
                similar_input[i] = evoked_indices[most_similar] if hamming_distances[i] < PARA.c.stats.hamming_threshold else -1

            #similar_input[i] = evoked_indices[most_similar]

        # If current state IS silent
        else:
            similar_input[i] = -1

    print(np.shape(similar_input))

    #return similar_input

