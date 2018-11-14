from evaluation import *

def plot_trace(activity):
    # activity: models, test chunks
    
    num_models = np.shape(activity)[0]
    cols = cm.rainbow(np.linspace(0, 1, num_models))
    test_steps = np.arange(np.shape(activity)[1]) * para.c.stats.transition_step_size + para.c.stats.transition_step_size
    
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
