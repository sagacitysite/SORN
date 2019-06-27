import evaluation as ev
from evaluation import *

def plot_trace(activity):
    # activity: models, test chunks
    
    num_models = np.shape(activity)[0]
    cols = cm.rainbow(np.linspace(0, 1, num_models))
    test_steps = np.arange(np.shape(activity)[1]) * PARA.c.stats.transition_step_size + PARA.c.stats.transition_step_size
    
    # Plot mean of every model
    for i in range(num_models):
        legend = 'Model ' + str(i + 1)
        plt.errorbar(test_steps, activity[i,:], label=legend, color=cols[i])

    # Beautify plot and save png file
    plt.legend(loc=4, prop={'size': legend_size})
    plt.ylim(ymin=0, ymax=np.max(activity)+0.02)
    plt.xlabel('Test steps', color=fig_color)
    plt.ylabel('Activity', color=fig_color)
    plt.savefig(plotpath + '/test_traces_activity.'+file_type, format=file_type, transparent=True)
    plt.close()

"""
@desc:   Checks how often a specific pattern (like 0,1,2,3) is contained in an arra
@params:
    arr: Array in which to search for
    seach_size: Length of pattern (3: 0,1,2; 4: 0,1,2,3; etc.)
"""
def count_pattern_in_sequence(arr, search_size):
    condition = (arr == 0)
    for i in range(1,search_size):
        condition = condition & (np.roll(arr,-i) == i)
    return np.sum(condition)

"""
@desc: Calculates number of correct sequences
"""
def count_sequences(activity):
    print('# Count ABCD sequences')
    # runs, models, chunks, steps per chunk

    # Remove first chunk
    activity = activity[:,:,1:,:]

    # Prepare some variables
    length_of_sequence = PARA.c.source.transitions.shape[1]
    num_models = PARA.c.source.transitions.shape[0]
    num_chunks = NUM_CHUNKS-1
    num_sequences_total = NUM_RUNS * num_chunks * PARA.c.stats.transition_step_size

    # Reshape array for further computation
    act = np.transpose(activity, (1, 0, 2, 3)).reshape((num_models, num_sequences_total))

    # Count number of patterns occuring for each model
    num_patterns = np.array([ count_pattern_in_sequence(act[i,:], length_of_sequence) for i in range(num_models) ])

    # Calculate relative result
    res = length_of_sequence*num_patterns/float(num_sequences_total)

    # Calculate entropies for models
    entropies = np.array([ev._helper.transition_entropy(t) for t in PARA.c.source.transitions])

    # Plot
    plt.plot(entropies, res)
    plt.ylim(ymin=0, ymax=1)
    plt.xlabel('Entropy of input while testing', color=FIG_COLOR)
    plt.ylabel('Frequency of correct patterns', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/frequency_of_pattern.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
