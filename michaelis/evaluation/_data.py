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
        noplastic_test = np.load(path + '/run' + str(i) + '.npy')
        # Calculate statistics for spontaneous activity
        activity.append(spont_activity(noplastic_test))

        # ...
    
    # Stack all statistics together
    spont_stats_all = np.stack(activity, axis=0)
    # ...

    print(np.shape(spont_stats_all))
    # probably: runs, chunks, models, train steps, thresholds, h_ip, #ee-connections, neurons

    return spont_stats_all

"""
@desc: Calculate statistics for one noplastic test run (sponaneous activity)
"""
def spont_activity(spont_spikes):
    transition_step_size = PARA.c.stats.transition_step_size
    spont_spikes_steps = PARA.c.steps_noplastic_test
    num_transition_steps = int(round((spont_spikes_steps / transition_step_size) - 0.5))

    spont_activity_chunks = [spont_spikes[:,:,:,:,:,:,i*transition_step_size:(i+1)*transition_step_size] for i in range(num_transition_steps)]
    spont_activity_chunks_mean = np.mean(spont_activity_chunks, axis=7)  # mean over chunks
    # chunks, x, x, x, x, x, neurons
    # probably: chunks, models, train steps, thresholds, h_ip, #ee-connections, neurons

    return spont_activity_chunks_mean

#def calc_activity():

