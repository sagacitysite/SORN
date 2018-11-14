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
