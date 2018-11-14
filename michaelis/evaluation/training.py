from evaluation import *

def plot_steps(distances, suffix, ytext, leg_loc=1):
    print('# Training: Plot steps, '+ytext)
    
    # runs, models, train steps, test chunks, dists
    # Get mean over "runs" and over "test chunks"
    # CAUTION: Mean over "test steps" is only appropriate if STDP is switched off in test phase
    dists_mean = np.mean(distances, axis=(0,3))
    dists_std = np.std(distances, axis=(0,3))

    # Mean just over test chunks, not over runs
    dists_mean_single = np.mean(distances, axis=3)

    # Get number of models and calculate train steps
    num_models = np.shape(distances)[1]

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, num_models))

    # Plot influence of training steps for every model
    for i in range(num_models):
        legend = 'Model '+str(i+1)
        plt.errorbar(PARA.c.steps_plastic, dists_mean[i,:], label=legend, yerr=dists_std[i], color=color_palette[i],
                     elinewidth=1, ecolor=np.append(color_palette[i][0:3], 0.5))

        # Plot all runs of every model (transparent in background)
        for j in range(np.shape(dists_mean_single)[0]):
            plt.plot(PARA.c.steps_plastic, dists_mean_single[j, i], color=color_palette[i], alpha=0.1)

    # Beautify plot and save png file
    plt.legend(loc=leg_loc, prop={'size': LEGEND_SIZE})
    plt.ylim(ymin=0)
    plt.xlabel('Training steps', color=FIG_COLOR)
    plt.ylabel(ytext, color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/distances_training_steps_'+suffix+'.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
    
def plot_thresholds(distances):
    print('# Training: Plot steps with threshold')
    
    # runs, models, train steps, thresholds, test steps
    
    num_test_chunks = np.shape(distances)[4]
    # Get last test chunkg and mean over runs
    dists_mean = np.mean(distances[:,:,:,:,num_test_chunks-1], axis=0)
    # Format after: models, train steps, thresholds

    # Get number of models and number of thresholds
    num_models = np.shape(distances)[1]
    num_thresh = np.shape(distances)[3]

    # Define color palette
    thresh_colors = cm.rainbow(np.linspace(0, 1, num_thresh))
    model_colors = cm.rainbow(np.linspace(0, 1, num_models))

    # Plot influence of training steps for every model
    for i in range(num_models):
        for j in range(num_thresh):
            opacity = 1 if j == num_thresh-1 else 0.2
            legend = None
            if j == 0:
                legend = 'Model '+str(i+1)+', Threshold finite'
            elif j == num_thresh-1:
                legend = 'Model '+str(i+1)+', Threshold inf'
            plt.errorbar(PARA.c.steps_plastic, dists_mean[i,:,j], label=legend, color=np.append(model_colors[i][0:3], opacity))
                         #path_effects=[pe.SimpleLineShadow(shadow_color=thresh_colors[j]), pe.Normal()])

    # Beautify plot and save png file
    leg = plt.legend(framealpha=0.8,prop={'size': LEGEND_SIZE})
    leg.get_frame().set_linewidth(0.0)
    plt.ylim(ymin=0)
    plt.xlabel('Training steps', color=FIG_COLOR)
    plt.ylabel('Transition error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/distances_training_steps_with_thresholds.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
