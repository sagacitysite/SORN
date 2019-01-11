import evaluation as ev
from evaluation import *

def barplot_weights_p1(errors, errors_sd):
    barplot_weights(errors, errors_sd, filename='error_p1_weights', ylabel='p1 error')

def barplot_weights_p2(errors, errors_sd):
    barplot_weights(errors, errors_sd, filename='error_p2_weights', ylabel='p2 error')

def barplot_weights(errors, errors_sd, filename, ylabel):
    # TODO Generalize for different training steps

    print('# Error Barplot: Weight clusters')
    # models, error

    # Initalize variables
    num_models = np.shape(errors)[0]
    x = np.arange(num_models)+1

    # Plot
    plt.bar(x, errors, 0.5, linewidth=0, yerr=errors_sd)
    plt.ylim(ymin=0)
    plt.xlabel('Model', color=FIG_COLOR)
    plt.ylabel(ylabel, color=FIG_COLOR)
    plt.savefig(PLOTPATH +'/'+ filename + '.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

def barplot_spont(errors, filename, ylabel):
    print('# Error Barplot: Spontaneous activity')
    # runs, models, train, error

    # Initalize dimensions
    num_models = np.shape(errors)[1]
    num_train_steps = np.shape(errors)[2]

    # Initalize data
    train_min = errors[:,:,0] if num_train_steps > 1 else None
    train_max = errors[:,:,-1]

    # Initalize plots
    shift = 1 if train_min is None else 0.75
    bar_width = 0.5 if train_min is None else 0.25
    x = np.arange(num_models) + shift

    # Plot errors for train max
    legend = None if train_min == None else str(PARA.c.steps_plastic[-1])+' training steps'
    plt.bar(x, np.mean(train_max, axis=0), bar_width, label=legend, linewidth=0, yerr=np.std(train_max, axis=0))

    # Plot errors for train min
    if train_min is not None:
        legend = str(PARA.c.steps_plastic[0])+' training steps'
        plt.bar(x, np.mean(train_min, axis=0), bar_width, color='red', label=legend, linewidth=0, yerr=np.std(train_min, axis=0), ecolor='red')
        plt.legend(prop={'size': LEGEND_SIZE})
    
    # Add decoration and save
    #if ymax:
    #    plt.ylim(ymax=ymax)
    plt.ylim(ymin=0)
    plt.xlabel('Model', color=FIG_COLOR)
    plt.ylabel(ylabel, color=FIG_COLOR)
    plt.savefig(PLOTPATH +'/'+ filename + '.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

def manual_plots():
    print('# Manual performance plot')

    # 3stat 4stat1 4stat2 5stat
    data = [
        { 'title': 'L1 error - Spontaneous activity', 'ylabel': 'L1 error', 'err': [0.161, 0.038, 0.024, 0.120]},
        { 'title': 'L2 error - Spontaneous activity', 'ylabel': 'L2 error', 'err': [0.185, 0.056, 0.034, 0.136]},
        { 'title': 'L1 error - Weight clusters', 'ylabel': 'L1 error', 'err': [0.059, 0.067, 0.095, 0.080]},
        { 'title': 'L2 error - Weight clusters', 'ylabel': 'L2 error', 'err': [0.064, 0.073, 0.112, 0.083]},
    ]

    ymax = np.ceil(np.max(np.array([d['err'] for d in data]).flatten())*100)/100

    x = np.arange(4) + 1
    bar_width = 0.5
    for i, d in enumerate(data):
        plt.bar(x, d['err'], bar_width, linewidth=0)
        plt.ylim(ymin=0,ymax=ymax)
        plt.title(d['title'])
        plt.xlabel('Model', color=FIG_COLOR)
        plt.ylabel(d['ylabel'], color=FIG_COLOR)
        plt.savefig(PLOTPATH + '/performance_'+ str(i+1) +'.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()

def errors_signalnoise(transitions):
    # Number of steps, where transition matrix is calculated for
    num_states = np.size(PARA.c.states)

    # Get transition matrices (inital and clusters)
    we = transitions
    # FIXME This is a fixed transition matrix for 4 states!
    it_signal = np.array([[0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [1, 0, 0, 0]])
    it_noise = np.array([[0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25]])

    # p2 matrix norm regarding signal
    p2_signal = np.sqrt(np.sum(np.square(we - it_signal), axis=(3,4)))
    p2_signal_mean = np.mean(p2_signal, axis=0)
    p2_signal_sd = np.std(p2_signal, axis=0)

    # p2 matrix norm regarding noise
    p2_noise = np.sqrt(np.sum(np.square(we - it_noise), axis=(3,4)))
    p2_noise_mean = np.mean(p2_noise, axis=0)
    p2_noise_sd = np.std(p2_noise, axis=0)

    return p2_signal_mean, p2_signal_sd, p2_noise_mean, p2_noise_sd
