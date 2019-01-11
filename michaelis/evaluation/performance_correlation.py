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
    plt.errorbar(variance, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0) , fmt='o')

    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('Variance/Distances')
    plt.xlabel('Variance', color=FIG_COLOR)
    plt.ylabel('Transition error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_spont_variance.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

    # Variance train
    for i in range(train_steps):
        legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(variance, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0), fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': LEGEND_SIZE})
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('Variance/Distances')
    plt.xlabel('Variance', color=FIG_COLOR)
    plt.ylabel('Transition error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_spont_variance_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

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
    if PARA.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
        dists_baseline = np.mean(dists[:, :, 0, hpos_idx], axis=0)
        for i in range(train_steps):
            if i > 0:
                # Variance with distance difference
                diff = dists_baseline.flatten() - np.mean(dists[:, :, i, hpos_idx], axis=0)
                legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, diff)[0], 2))
                plt.plot(variance, diff, label=legend, color=train_colors[i])

        plt.legend(prop={'size': LEGEND_SIZE})
        plt.ylim(ymin=0)
        #plt.grid()
        #plt.title('Baseline: Variance/Distances')
        plt.xlabel('Variance', color=FIG_COLOR)
        plt.ylabel('Performance increase in relation to baseline', color=FIG_COLOR)
        plt.savefig(PLOTPATH + '/correlation_spont_variance_train_baseline.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()

    # KL (highest train, standard h_ip)
    plt.errorbar(kl, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0), fmt='o')
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('KL/Distances')
    plt.xlabel('KL', color=FIG_COLOR)
    plt.ylabel('Transition error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_spont_kl.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

    # KL train
    for i in range(train_steps):
        legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(kl, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(kl, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': LEGEND_SIZE})
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('KL/Distances')
    plt.xlabel('KL', color=FIG_COLOR)
    plt.ylabel('Transition error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_spont_kl_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
    
    # Transition entropy (highest train, standard h_ip)
    plt.errorbar(transition_entropy, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0), fmt='o')
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('KL/Distances')
    plt.xlabel('entropy rate', color=FIG_COLOR)
    plt.ylabel('p2 error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_spont_entropy.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
    
    # Transition entropy train
    for i in range(train_steps):
        legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(kl, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(transition_entropy, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend,
                     yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': LEGEND_SIZE})
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('KL/Distances')
    plt.xlabel('entropy rate', color=FIG_COLOR)
    plt.ylabel('p2 error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_spont_entropy_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

    # KL baseline plot
    if PARA.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
        dists_baseline = np.mean(dists[:, :, 0, hpos_idx], axis=0)
        for i in range(train_steps):
            if i > 0:
                # Variance with distance difference
                diff = dists_baseline - np.mean(dists[:, :, i,hpos_idx], axis=0)
                legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(kl, diff)[0], 2))
                plt.plot(kl, diff, label=legend, color=train_colors[i])

        plt.legend(prop={'size': LEGEND_SIZE})
        plt.ylim(ymin=0)
        #plt.grid()
        #plt.title('Baseline: KL/Distances')
        plt.xlabel('KL', color=FIG_COLOR)
        plt.ylabel('Performance increase in relation to baseline', color=FIG_COLOR)
        plt.savefig(PLOTPATH + '/correlation_spont_kl_train_baseline.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
        plt.close()

    # Gini
    for i in range(train_steps):
        legend = str(PARA.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(ginis, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(ginis, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': LEGEND_SIZE})
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('Gini/Distances')
    plt.xlabel('Gini', color=FIG_COLOR)
    plt.ylabel('Transition error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_spont_gini_train.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

def plot_entropy_weights(errors, std):
    # models, errors

    transition_entropy = np.array([ev._helper.transition_entropy(t) for t in PARA.c.source.transitions])

    # Transition entropy (highest train, standard h_ip)
    plt.errorbar(transition_entropy, errors, yerr=std, fmt='o')
    plt.ylim(ymin=0)
    plt.xlabel('entropy rate', color=FIG_COLOR)
    plt.ylabel('p2 error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/correlation_weights_entropy.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

def plot_entropy_signalnoise(errors_signal, std_signal, errors_noise, std_noise, filename):
    # models, errors
    
    # Calculate entropy
    transition_entropy = np.array([ev._helper.transition_entropy(t) for t in PARA.c.source.transitions])

    # Plot
    plt.errorbar(transition_entropy, errors_signal, yerr=std_signal, label='Signal error', fmt='o')
    plt.errorbar(transition_entropy, errors_noise, yerr=std_noise, label='Noise error', fmt='o')
    plt.legend(prop={'size': LEGEND_SIZE})
    plt.ylim(ymin=0)
    plt.xlabel('entropy rate', color=FIG_COLOR)
    plt.ylabel('p2 error regarding signal/noise', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/'+ filename +'.'+ FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()

def plot_entropy_signalnoise_weights(errors_signal, std_signal, errors_noise, std_noise):
    plot_entropy_signalnoise(errors_signal, std_signal, errors_noise, std_noise, filename="correlation_weights_entropy_signalnoise")

def plot_entropy_signalnoise_spont(errors_signal, std_signal, errors_noise, std_noise):
    plot_entropy_signalnoise(errors_signal, std_signal, errors_noise, std_noise, filename="correlation_spont_entropy_signalnoise")
