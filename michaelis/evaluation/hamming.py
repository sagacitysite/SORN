from evaluation import *

def plot_histogram(hamming_distances):
    #global plotpath

    # Mean over runs
    hamming_mean = np.mean(hamming_distances, axis=0)
    # Form after: models, train steps, hammings

    # Get largest, middle and smallest training run
    hms = np.shape(hamming_mean)
    hmean_maxmin = np.empty((hms[0], 3, hms[2]))
    maxmin_idx = np.array([0, (hms[1]-1)/2, hms[1]-1])
    hmean_maxmin[:, 0, :] = hamming_mean[:, maxmin_idx[0], :]
    hmean_maxmin[:, 1, :] = hamming_mean[:, maxmin_idx[1], :]
    hmean_maxmin[:, 2, :] = hamming_mean[:, maxmin_idx[2], :]
    # Form after: models, min/middle/max train steps, hammings

    # Get overal max hamming value and frequency, used for axis
    max_freq = np.max([np.max(np.bincount(hmean_maxmin[i, j, :].astype(int))) for j in range(3) for i in range(hms[0])])
    max_val = np.max(hmean_maxmin)

    # Barplot preparation
    bar_width = 0.15  # width of bar
    intra_bar_space = 0.05  # space between multi bars
    coeff = np.array([-1, 0, 1])  # left, middle, right from tick

    # Define color palette
    color_palette = cm.copper(np.linspace(0, 1, 3))

    # Check normality (TODO already calculated, can be used if necessary)
    #pvalues = [scipy.stats.kstest(hmean_maxmin[i, j, :], 'norm')[0] for j in range(3) for i in range(hms[0])]

    for i in range(hms[0]):
        for j in range(3):
            # Calculate frequencies
            hamming_freqs = np.bincount(hmean_maxmin[i, j, :].astype(int))
            # Calculate x positions
            x = np.arange(len(hamming_freqs)) + (coeff[j] * (bar_width + intra_bar_space))
            # Plot barplot with legend
            legend = str(para.c.steps_plastic[maxmin_idx[j]]) + ' training steps'
            plt.bar(x, hamming_freqs, bar_width, label=legend, linewidth=0, color=color_palette[j])
        # Adjust plot
        plt.ylim(ymax=max_freq)
        plt.xlim(xmin=0, xmax=max_val)
        plt.legend(prop={'size': legend_size})
        #plt.title('Model: ' + str(i+1))
        plt.xlabel('Hamming distance', color=fig_color)
        plt.ylabel('Frequency', color=fig_color)
        plt.savefig(plotpath + '/hamming_freqs_model'+str(i+1)+'.'+file_type, format=file_type, transparent=True)
        plt.close()