from evaluation import *

def plot(distances, suffix, ylabel, title=None, ymax=None):
    print('# Test traces: Plot, '+ylabel)
    
    #global plotpath, para
    # runs, models, train steps, test steps / test chunks

    train_max = distances[:, :, np.shape(distances)[2]-1]
    # runs, models, test chunks

    # If there is more than one training step, also evaluate train_min as baseline
    train_min = None
    if np.shape(distances)[2] > 1:
        train_min = distances[:, :, 0]
        # runs, models, test chunks

    # Calculate mean and standard deviation
    dists_mean = np.mean(train_max, axis=0)
    dists_std = np.std(train_max, axis=0)

    # Get number of original test steps (for x axis)
    test_steps = np.arange(np.shape(train_max)[2]) * PARA.c.stats.transition_step_size + PARA.c.stats.transition_step_size
    num_models = np.shape(train_max)[1]
    test_chunks = np.shape(train_max)[2]

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, np.shape(dists_mean)[0]))

    # t-Tests
    file = ""
    for i in range(num_models):
        for j in range(num_models):
            if j > i:
                res =scipy.stats.ttest_ind(train_max[:,i,test_chunks-1], train_max[:,j,test_chunks-1])
                file += "Model "+str(i+1)+" vs. "+str(j+1)+": t="+str(res[0])+", p="+str(res[1])+"\n"
    text_file = open(PLOTPATH + "/t-tests_"+suffix+".txt", "w")
    text_file.write(file)
    text_file.close()

    # Plot mean of every model
    for i in range(num_models):
        legend = 'Model ' + str(i + 1)
        plt.errorbar(test_steps, dists_mean[i,:], label=legend, yerr=dists_std[i], color=color_palette[i],
                     elinewidth=1, ecolor=np.append(color_palette[i][0:3], 0.5))

        # Plot all runs of every model (transparent in background)
        for j in range(np.shape(train_max)[0]):
            plt.plot(test_steps, train_max[j,i], color=color_palette[i], alpha=0.1)

    # Beautify plot and save png file
    plt.legend(prop={'size': LEGEND_SIZE})
    if title:
        plt.title(title)
    if ymax:
        plt.ylim(ymax=ymax)
    plt.ylim(ymin=0)
    plt.xlabel('Test steps', color=FIG_COLOR)
    plt.ylabel(ylabel, color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/test_traces_'+suffix+'.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
