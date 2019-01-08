import evaluation as ev
from evaluation import *

def plot_learned(learned_transitions):
    # runs, models, matrix
    print('# Transitions: Plot learned transition matrix')

    # Mean over runs
    transitions = np.mean(learned_transitions, axis=0)

    # Plot transitions matrices for all models
    plot_transition_matrices(transitions, 'transitions-learned')

def plot_initial():
    print('# Transitions: Plot initial transition matrix')

    # Get initial transitions from parameter file and plot for all models
    plot_transition_matrices(PARA.c.source.transitions, 'transitions-initial')
    
def plot_transition_matrices(matrices, filename):

    # Loop over models
    for i in range(np.shape(matrices)[0]): # num_models
        # Get transition matrix of current model
        matrix = matrices[i,:,:]

        # Prepare plot parameters
        num_ticks = np.arange(np.shape(matrix)[0])
        ticks = PARA.c.states

        # Plot transition matrix
        plt.imshow(matrix, clim=(0,0.5), origin='upper', cmap='copper_r', interpolation='nearest')
        for (k,l),label in np.ndenumerate(matrix):
            col = 'black' if matrix[l,k] < 0.25 else 'white'
            plt.text(k, l, np.around(matrix[l,k], 2), color=col, size=11, ha='center', va='center')
        plt.xlabel('To')
        plt.xticks(num_ticks, ticks)
        plt.ylabel('From')
        plt.yticks(num_ticks, ticks)
        plt.colorbar()
        #plt.title('Model '+str(i+1))
        plt.savefig(PLOTPATH +'/'+ filename +'_model'+ str(i+1) +'.'+ FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()
