import evaluation as ev
from evaluation import *

def plot_activity(distances, activity):
    print('# Performance correlation: Activity plot')
    #global PLOTPATH, test_step_size

    # Get results for highest train step only
    last_idx_train_steps = np.shape(distances)[2] - 1
    dists = distances[:, :, last_idx_train_steps, :]
    actis = activity[:, :, last_idx_train_steps, :]

    # x = np.mean(dists[:,:,:], axis=(0,2)).flatten()
    # y = np.mean(actis[:,:,:], axis=(0,2)).flatten()
    # plt.scatter(x, y)

    for i in range(np.shape(dists)[1]):
        plt.figure()
        x = dists[:,i,:].flatten()
        y = actis[:,i,:].flatten()
        plt.scatter(x, y)
        #plt.title('Correlation Activity/Distances (%.2f)' % pearsonr(x, y)[0])
        plt.xlabel('Transition error', color=FIG_COLOR)
        plt.ylabel('Activity (percentage)', color=FIG_COLOR)
        plt.savefig(PLOTPATH + '/correlation_activity_model' + str(i+1) + '.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()

def plot_ncomparison(distances, ncomparison):
    print('# Performance correlation: nComparison plot')
    #global PLOTPATH, test_step_size

    # Get results for highest train step only
    last_idx_train_steps = np.shape(distances)[2] - 1

    dists0 = distances[:, :, 0, :]
    ncomp0 = ncomparison[:, :, 0]

    dists = distances[:, :, last_idx_train_steps, :]
    ncomp = ncomparison[:, :, last_idx_train_steps]

    num_models = np.shape(dists)[1]

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, num_models))

    # Plot (no train = static)
    plt.figure()
    for i in range(num_models):
        x = np.mean(dists0[:, i, 1:], axis=1) # Exclude first test step
        y = ncomp0[:, i]
        plt.scatter(x, y, color=color_palette[i], alpha=0.3)
        legend = 'Model ' + str(i+1)
        plt.errorbar(np.mean(x), np.mean(y), yerr=np.std(y), label=legend, fmt='o', color=color_palette[i])

    # Add decoration
    #plt.title('Correlation NComparison/Distances (%.2f)' % pearsonr(np.mean(ncomp, axis=0), np.mean(dists, axis=(0,2)))[0])
    #print('Ncomparison correlation static: '+str(pearsonr(np.mean(ncomp0, axis=0), np.mean(dists0, axis=(0,2)))[0]))
    plt.xlabel('Transition error', color=FIG_COLOR)
    plt.ylabel('Number of comparison states', color=FIG_COLOR)
    plt.legend(prop={'size': legend_size})
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=3000)

    # Save and close plit
    plt.savefig(PLOTPATH + '/correlation_ncomparison_distances_static.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

    # Plot (train)
    plt.figure()
    for i in range(num_models):
        x = np.mean(dists[:, i, 1:], axis=1) # Exclude first test step
        y = ncomp[:, i]
        legend = 'Model ' + str(i+1)
        plt.scatter(x, y, color=color_palette[i], alpha=0.3)
        plt.errorbar(np.mean(x), np.mean(y), yerr=np.std(y), label=legend, fmt='o', color=color_palette[i])

    # Add decoration
    #plt.title('Correlation NComparison/Distances (%.2f)' % pearsonr(np.mean(ncomp, axis=0), np.mean(dists, axis=(0,2)))[0])
    #print('Ncomparison correlation: '+str(pearsonr(np.mean(ncomp, axis=0), np.mean(dists, axis=(0,2)))[0]))
    plt.xlabel('Transition error', color=FIG_COLOR)
    plt.ylabel('Number of comparison states', color=FIG_COLOR)
    plt.legend(prop={'size': legend_size})
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=3000)

    # Save and close plit
    plt.savefig(PLOTPATH + '/correlation_ncomparison_distances.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
    
def plot_inequality(distances, hpos_idx):
    print('# Performance correlation: Inequality plot')
    
    # runs, models, train steps, h_ip, test chunks

    # Calculate stationary distributions from markov chains
    stationaries = np.array([ev._helper.stationary_distribution(transition) for transition in PARA.c.source.transitions])

    # Get variance, entropy and gini
    states = np.arange(np.shape(stationaries)[1])+1
    variance = np.var(stationaries, axis=1)
    variance[variance < 1e-16] = 0
    #variance = np.sum(np.multiply(stationaries, (states - np.mean(states)) ** 2), axis=1)
    kl = np.array([scipy.stats.entropy(s, np.repeat(0.25, np.shape(PARA.c.source.transitions)[1])) for s in stationaries])
    kl[kl < 1e-16] = 0
    transition_entropy = np.array([ev._helper.transition_entropy(t) for t in PARA.c.source.transitions])
    ginis = np.array([ev._helper.calc_gini(x) for x in stationaries])
    ginis[ginis < 1e-15] = 0
    traces = np.array([np.trace(t) for t in PARA.c.source.transitions])

    # Get number of train steps
    num_models = np.shape(distances)[1]
    train_steps = np.shape(distances)[2]
    hip_steps = np.shape(distances)[3]

    # Exclude first test step and mean over test steps
    dists = np.mean(distances[:, :, :, :, 1:], axis=4)

    # Define color palette
    if train_steps > 2:
        train_colors = cm.rainbow(np.linspace(0, 1, train_steps))
    elif train_steps == 2:
        train_colors = [(1,0,0), (0,0,1)]  # red, blue
    else:
        train_colors = [(0,0,1)]  # blue

    hip_colors = cm.rainbow(np.linspace(0, 1, hip_steps))

    # Variance (highest train, standard h_ip)
    # plt.errorbar(variance, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
    #              yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0) , fmt='o')

    # plt.ylim(ymin=0)
    # #plt.grid()
    # #plt.title('Variance/Distances')
    # plt.xlabel('Variance', color=FIG_COLOR)
    # plt.ylabel('Transition error', color=FIG_COLOR)
    # plt.savefig(PLOTPATH + '/correlation_spont_variance.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    # plt.close()

    # Variance train
    # for i in range(train_steps):
    #     legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
    #     plt.errorbar(variance, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0), fmt='o',
    #                  color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    # plt.legend(loc=2,prop={'size': LEGEND_SIZE})
    # plt.ylim(ymin=0)
    # #plt.grid()
    # #plt.title('Variance/Distances')
    # plt.xlabel('Variance', color=FIG_COLOR)
    # plt.ylabel('Transition error', color=FIG_COLOR)
    # plt.savefig(PLOTPATH + '/correlation_spont_variance_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    # plt.close()

    # Variance h_ip
    #hips = None
    #if ip == 'h_ip_factor':
    #    hips = np.round(np.mean(PARA.c[ip], axis=0), 3)
    #else:
    #    hips = PARA.c[ip]

    #if len(hips) > 1:
    #    for i in range(hip_steps):
    #        errors = np.mean(dists[:,:,train_steps-1,i], axis=0)
    #        legend = str(hips[i]) + ' ' + str(ip) + ', abs=' + str(np.round(np.abs(np.max(errors) - np.min(errors)), 3))
    #        plt.errorbar(variance, np.mean(dists[:,:,train_steps-1,i], axis=0), label=legend, yerr=np.std(dists[:,:,train_steps-1,i], axis=0),  fmt='o',
    #                     color=hip_colors[i], ecolor=np.append(hip_colors[i][0:3], 0.5))

    #    plt.legend(loc=2, prop={'size': legend_size})
    #    plt.ylim(ymin=0)
    #    #plt.grid()
    #    #plt.title('Variance/Distances')
    #    plt.xlabel('Variance', color=FIG_COLOR)
    #    plt.ylabel('Transition error', color=FIG_COLOR)
    #    plt.savefig(PLOTPATH + '/correlation_spont_variance_hip.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    #    plt.close()

    # Variance baseline plot
    # if PARA.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
    #     dists_baseline = np.mean(dists[:, :, 0, hpos_idx], axis=0)
    #     for i in range(train_steps):
    #         if i > 0:
    #             # Variance with distance difference
    #             diff = dists_baseline.flatten() - np.mean(dists[:, :, i, hpos_idx], axis=0)
    #             legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, diff)[0], 2))
    #             plt.plot(variance, diff, label=legend, color=train_colors[i])

    #     plt.legend(prop={'size': LEGEND_SIZE})
    #     plt.ylim(ymin=0)
    #     #plt.grid()
    #     #plt.title('Baseline: Variance/Distances')
    #     plt.xlabel('Variance', color=FIG_COLOR)
    #     plt.ylabel('Performance increase in relation to baseline', color=FIG_COLOR)
    #     plt.savefig(PLOTPATH + '/correlation_spont_variance_train_baseline.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    #     plt.close()

    # KL (highest train, standard h_ip)
    # plt.errorbar(kl, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
    #              yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0), fmt='o')
    # plt.ylim(ymin=0)
    # #plt.grid()
    # #plt.title('KL/Distances')
    # plt.xlabel('KL', color=FIG_COLOR)
    # plt.ylabel('Transition error', color=FIG_COLOR)
    # plt.savefig(PLOTPATH + '/correlation_spont_kl.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    # plt.close()

    # KL train
    # for i in range(train_steps):
    #     legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(kl, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
    #     plt.errorbar(kl, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
    #                  color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    # plt.legend(loc=2,prop={'size': LEGEND_SIZE})
    # plt.ylim(ymin=0)
    # #plt.grid()
    # #plt.title('KL/Distances')
    # plt.xlabel('KL', color=FIG_COLOR)
    # plt.ylabel('Transition error', color=FIG_COLOR)
    # plt.savefig(PLOTPATH + '/correlation_spont_kl_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    # plt.close()
    
    # Entropy rate (highest train, standard h_ip)
    plt.errorbar(transition_entropy, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0), fmt='o')
    plt.ylim(ymin=0)
    #plt.grid()
    plt.xlabel('entropy rate', color=FIG_COLOR)
    plt.ylabel('p2 error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_spont_entropy.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
    
    # Entropy rate train
    if train_steps > 1:
        for i in range(train_steps):
            legend = 'static reservoir' if ((i == 0) and (PARA.c.steps_plastic[0] == 0)) else str(PARA.c.steps_plastic[i]) + ' training steps'
            plt.errorbar(transition_entropy, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend,
                        yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
                        color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

        plt.legend(prop={'size': LEGEND_SIZE})
        plt.ylim(ymin=0)
        #plt.grid()
        plt.xlabel('entropy rate', color=FIG_COLOR)
        plt.ylabel('p2 error', color=FIG_COLOR)
        plt.savefig(PLOTPATH + '/correlation_spont_entropy_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()

    # Entropy rate baseline (highest train, compared to static)
    if (train_steps > 1) and (PARA.c.steps_plastic[0] == 0):
        p2_diff = np.abs(dists[:,:,train_steps-1,hpos_idx] - dists[:,:,0,hpos_idx])
        plt.errorbar(transition_entropy, np.mean(p2_diff, axis=0), yerr=np.std(p2_diff, axis=0), fmt='o')
        plt.ylim(ymin=0)
        #plt.grid()
        plt.xlabel('entropy rate', color=FIG_COLOR)
        plt.ylabel('p2 error difference', color=FIG_COLOR)
        plt.savefig(PLOTPATH + '/correlation_spont_entropy_static_baseline.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()

    # KL baseline plot
    # if PARA.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
    #     dists_baseline = np.mean(dists[:, :, 0, hpos_idx], axis=0)
    #     for i in range(train_steps):
    #         if i > 0:
    #             # Variance with distance difference
    #             diff = dists_baseline - np.mean(dists[:, :, i,hpos_idx], axis=0)
    #             legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(kl, diff)[0], 2))
    #             plt.plot(kl, diff, label=legend, color=train_colors[i])

    #     plt.legend(prop={'size': LEGEND_SIZE})
    #     plt.ylim(ymin=0)
    #     #plt.grid()
    #     #plt.title('Baseline: KL/Distances')
    #     plt.xlabel('KL', color=FIG_COLOR)
    #     plt.ylabel('Performance increase in relation to baseline', color=FIG_COLOR)
    #     plt.savefig(PLOTPATH + '/correlation_spont_kl_train_baseline.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    #     plt.close()

    # Gini
    # for i in range(train_steps):
    #     legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(ginis, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
    #     plt.errorbar(ginis, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
    #                  color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    # plt.legend(loc=2,prop={'size': LEGEND_SIZE})
    # plt.ylim(ymin=0)
    # #plt.grid()
    # #plt.title('Gini/Distances')
    # plt.xlabel('Gini', color=FIG_COLOR)
    # plt.ylabel('Transition error', color=FIG_COLOR)
    # plt.savefig(PLOTPATH + '/correlation_spont_gini_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    # plt.close()

def plot_entropy_weights(errors, std):
    # models, train, errors

    num_train_steps = np.shape(errors)[1]

    transition_entropy = np.array([ev._helper.transition_entropy(t) for t in PARA.c.source.transitions])

    # Define color palette
    if num_train_steps > 2:
        train_colors = cm.rainbow(np.linspace(0, 1, train_steps))
    elif num_train_steps == 2:
        train_colors = [(1,0,0), (0,0,1)]  # red, blue
    else:
        train_colors = [(0,0,1)]  # blue

    # Entropy rate (highest train, standard h_ip)
    plt.errorbar(transition_entropy, errors[:,-1], yerr=std[:,-1], fmt='o')
    plt.ylim(ymin=0)
    plt.xlabel('entropy rate', color=FIG_COLOR)
    plt.ylabel('p2 error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_weights_entropy.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

    # Entropy rate train
    if num_train_steps > 1:
        for i in range(num_train_steps):
            legend = 'static reservoir' if ((i == 0) and (PARA.c.steps_plastic[0] == 0)) else str(PARA.c.steps_plastic[i]) + ' training steps'
            plt.errorbar(transition_entropy, errors[:,i], yerr=std[:,i], fmt='o', label=legend,
                         color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

        plt.ylim(ymin=0)
        plt.legend(prop={'size': LEGEND_SIZE})
        plt.xlabel('entropy rate', color=FIG_COLOR)
        plt.ylabel('p2 error', color=FIG_COLOR)
        plt.savefig(PLOTPATH + '/correlation_weights_entropy_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()
    
    # TODO use raw data and mean over it afterwards, in order to get std
    # Entropy rate baseline (highest train, compared to static)
    # if (num_train_steps > 1) and (PARA.c.steps_plastic[0] == 0):
    #     errors_diff = np.abs(errors[:,-1] - errors[:,0])
    #     plt.errorbar(transition_entropy, errors[:,-1], yerr=std[:,-1], fmt='o')
    #     plt.ylim(ymin=0)
    #     plt.xlabel('entropy rate', color=FIG_COLOR)
    #     plt.ylabel('p2 error', color=FIG_COLOR)
    #     plt.savefig(PLOTPATH + '/correlation_weights_entropy_static_baseline.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    #     plt.close()

def plot_entropy_signalnoise(errors_signal, std_signal, errors_noise, std_noise, filename):
    # models, errors
    
    # Calculate entropy
    transition_entropy = np.array([ev._helper.transition_entropy(t) for t in PARA.c.source.transitions])

    # Calculate theoretical p2 errors
    it_signal = np.array([[0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])
    transitions = []
    iterate = np.arange(0, 0.25, 0.001)
    for it in iterate:
        transitions.append([[it, 1.-(3*it), it, it],
                            [it, it, 1.-(3*it), it],
                            [it, it, it, 1.-(3*it)],
                            [1.-(3*it), it, it, it]])
    transitions = np.array(transitions)

    # TODO calculate entropies of 'transitions'
    th_entropy = np.array([ev._helper.transition_entropy(t) for t in transitions])
    th_errors = np.sqrt(np.sum(np.square(transitions - it_signal), axis=(1,2)))


    # Plot
    plt.plot(th_entropy, th_errors, color='darkorange')
    #plt.errorbar(transition_entropy, errors_signal[:,-1], yerr=std_signal[:,-1], label='Plastic', fmt='o')
    #plt.errorbar(transition_entropy, errors_signal[:,0], yerr=std_signal[:,0], label='Static', fmt='o')
    plt.errorbar(transition_entropy, errors_signal[:,-1], yerr=std_signal[:,-1], fmt='-o', markersize=4, capsize=4)
    #plt.errorbar(transition_entropy, errors_signal, yerr=std_signal, label='Signal error', fmt='o')
    #plt.errorbar(transition_entropy, errors_noise, yerr=std_noise, label='Noise error', fmt='o')
    #plt.legend(prop={'size': LEGEND_SIZE})
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel(r'$H_{input}$', color=FIG_COLOR)
    plt.ylabel(r'$\varepsilon_{signal}$', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/'+ filename +'.'+ FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

def plot_entropy_signalnoise_weights(errors_signal, std_signal, errors_noise, std_noise):
    plot_entropy_signalnoise(errors_signal, std_signal, errors_noise, std_noise, filename="correlation_weights_entropy_signalnoise")

def plot_entropy_signalnoise_spont(errors_signal, std_signal, errors_noise, std_noise):
    plot_entropy_signalnoise(errors_signal, std_signal, errors_noise, std_noise, filename="correlation_spont_entropy_signalnoise")

def plot_human_experiment():
    datapath = os.getcwd()
    exp = np.load(datapath+'/data/human/exp_data.npy')

    stepSize = 0.05
    maxEntropy = 1.41 # get_entropy(0.25) ~ 1.38
    bins = np.arange(0, maxEntropy, stepSize)

    # Number of trials
    performance = []
    times = []
    for i in range(bins.shape[0]-1):
        # Define start and end point to search for noise values
        fr, to = bins[i], bins[i+1]
        # Get temporarary part of array where noise lies in range
        tmp = exp[(exp[:,7] >= fr) & (exp[:,7] < to)]
        # Get number of trials in chosen range
        numTrials = tmp.shape[0]
        #print(numTrials)
        # Calculate performance
        performance.append(np.sum(tmp[:,3] == tmp[:,6]).astype(float)/numTrials)
        # Calculate mean of reaction time
        times.append(np.mean(tmp[:,4]))

    # Performance
    performance = np.array(performance)

    # Take time in seconds
    times = np.array(times)/1000

    # Define values for x axis
    x = bins[:-1] + stepSize/2

    plt.plot(x, 1-performance, marker="o", markersize=4)
    plt.ylabel(r'$\varepsilon_{human}$')
    plt.xlabel(r'$H_{input}$')
    plt.xlim((-0.01,maxEntropy))
    plt.ylim((-0.05,0.55))
    plt.savefig(PLOTPATH + '/error_human.' + FILE_TYPE, format=FILE_TYPE, transparent=True)
