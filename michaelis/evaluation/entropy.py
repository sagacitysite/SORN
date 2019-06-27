import evaluation as ev
from evaluation import *

"""
@desc: Entropy-Entropy plot
"""
def plot(x, y, xlabel, ylabel, filename, title=None):
    # Calculate entropy of x
    x_entropies = np.array([ev._helper.transition_entropy(t) for t in x])
    # Calculate entropy of y
    y_entropies = np.array([ np.array([ev._helper.transition_entropy(t) for t in y[i]]) for i in range(NUM_RUNS)])

    # Define mean and std of y entropies
    y_mean = np.mean(y_entropies, axis=0)
    y_std = np.std(y_entropies, axis=0)

    plt.plot(np.arange(0, 1.5, 0.1), np.arange(0, 1.5, 0.1), color='darkorange')
    #plt.scatter(x_entropies, y_entropies, color="red")
    plt.errorbar(x_entropies, y_mean, yerr=y_std, fmt="-o", markersize=4, capsize=4)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(PLOTPATH + '/'+ filename +'.'+ FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

def initial_weights(weight_transitions):
    plot(PARA.c.source.transitions, weight_transitions,
         xlabel=r'$H_{input}$', ylabel=r'$H_{weight\;output}$',
         filename="entropy_weights-initial")

def initial_spont(spont_transitions):
    plot(PARA.c.source.transitions, spont_transitions,
         xlabel=r'$H_{input}$', ylabel=r'$H_{spont\;output}$',
         filename="entropy_spont-initial")
