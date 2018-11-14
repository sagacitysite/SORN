from evaluation import *

def plot_activity(distances, activity):
    #global plotpath, test_step_size

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
        plt.xlabel('Transition error', color=fig_color)
        plt.ylabel('Activity (percentage)', color=fig_color)
        plt.savefig(plotpath + '/correlation_activity_model' + str(i+1) + '.'+file_type, format=file_type, transparent=True)
        plt.close()

def plot_ncomparison(distances, ncomparison):
    #global plotpath, test_step_size

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
    plt.xlabel('Transition error', color=fig_color)
    plt.ylabel('Number of comparison states', color=fig_color)
    plt.legend(prop={'size': legend_size})
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=3000)

    # Save and close plit
    plt.savefig(plotpath + '/correlation_ncomparison_distances_static.'+file_type, format=file_type, transparent=True)
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
    plt.xlabel('Transition error', color=fig_color)
    plt.ylabel('Number of comparison states', color=fig_color)
    plt.legend(prop={'size': legend_size})
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=3000)

    # Save and close plit
    plt.savefig(plotpath + '/correlation_ncomparison_distances.'+file_type, format=file_type, transparent=True)
    plt.close()
    
def plot_inequality(distances, hpos_idx):
    #global plotpath, para
    
    # runs, models, train steps, h_ip, test chunks

    # Calculate stationary distributions from markov chains
    stationaries = np.array([stationaryDistribution.calculate(transition) for transition in para.c.source.transitions])

    # Get variance, entropy and gini
    states = np.arange(np.shape(stationaries)[1])+1
    variance = np.var(stationaries, axis=1)
    variance[variance < 1e-16] = 0
    #variance = np.sum(np.multiply(stationaries, (states - np.mean(states)) ** 2), axis=1)
    kl = np.array([scipy.stats.entropy(s, np.repeat(0.25, np.shape(para.c.source.transitions)[1])) for s in stationaries])
    kl[kl < 1e-16] = 0
    transition_entropy = np.array([stationaryDistribution.transition_entropy(t) for t in para.c.source.transitions])
    ginis = np.array([calc_gini(x) for x in stationaries])
    ginis[ginis < 1e-15] = 0
    traces = np.array([np.trace(t) for t in para.c.source.transitions])

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

    # Trace (highest train, standard h_ip)
    plt.errorbar(traces, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0), fmt='o')

    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('Traces/Distances')
    plt.xlabel('Trace', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_trace.'+file_type, format=file_type, transparent=True)
    plt.close()

    # Variance (highest train, standard h_ip)
    plt.errorbar(variance, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0) , fmt='o')

    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('Variance/Distances')
    plt.xlabel('Variance', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_variance.'+file_type, format=file_type, transparent=True)
    plt.close()

    # Variance train
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(variance, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0), fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': legend_size})
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('Variance/Distances')
    plt.xlabel('Variance', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_variance_train.'+file_type, format=file_type, transparent=True)
    plt.close()

    # Variance h_ip
    #hips = None
    #if ip == 'h_ip_factor':
    #    hips = np.round(np.mean(para.c[ip], axis=0), 3)
    #else:
    #    hips = para.c[ip]

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
    #    plt.xlabel('Variance', color=fig_color)
    #    plt.ylabel('Transition error', color=fig_color)
    #    plt.savefig(plotpath + '/correlation_inequality_variance_hip.'+file_type, format=file_type, transparent=True)
    #    plt.close()

    # Variance baseline plot
    if para.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
        dists_baseline = np.mean(dists[:, :, 0, hpos_idx], axis=0)
        for i in range(train_steps):
            if i > 0:
                # Variance with distance difference
                diff = dists_baseline.flatten() - np.mean(dists[:, :, i, hpos_idx], axis=0)
                legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(variance, diff)[0], 2))
                plt.plot(variance, diff, label=legend, color=train_colors[i])

        plt.legend(prop={'size': legend_size})
        plt.ylim(ymin=0)
        #plt.grid()
        #plt.title('Baseline: Variance/Distances')
        plt.xlabel('Variance', color=fig_color)
        plt.ylabel('Performance increase in relation to baseline', color=fig_color)
        plt.savefig(plotpath + '/correlation_inequality_variance_train_baseline.'+file_type, format=file_type, transparent=True)
        plt.close()

    # KL (highest train, standard h_ip)
    plt.errorbar(kl, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0), fmt='o')
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('KL/Distances')
    plt.xlabel('KL', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_kl.'+file_type, format=file_type, transparent=True)
    plt.close()

    # KL train
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(kl, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(kl, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': legend_size})
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('KL/Distances')
    plt.xlabel('KL', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_kl_train.'+file_type, format=file_type, transparent=True)
    plt.close()
    
    # Transition entropy (highest train, standard h_ip)
    plt.errorbar(transition_entropy, np.mean(dists[:, :, train_steps-1, hpos_idx], axis=0),
                 yerr=np.std(dists[:, :, train_steps-1, hpos_idx], axis=0), fmt='o')
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('KL/Distances')
    plt.xlabel('entropy', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_entropy.'+file_type, format=file_type, transparent=True)
    plt.close()
    
    # Transition entropy train
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(kl, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(transition_entropy, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend,
                     yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': legend_size})
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('KL/Distances')
    plt.xlabel('entropy', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_entropy_train.'+file_type, format=file_type, transparent=True)
    plt.close()

    # KL baseline plot
    if para.c.steps_plastic[0] == 0:  # only if first training step is zero (baseline)
        dists_baseline = np.mean(dists[:, :, 0, hpos_idx], axis=0)
        for i in range(train_steps):
            if i > 0:
                # Variance with distance difference
                diff = dists_baseline - np.mean(dists[:, :, i,hpos_idx], axis=0)
                legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(kl, diff)[0], 2))
                plt.plot(kl, diff, label=legend, color=train_colors[i])

        plt.legend(prop={'size': legend_size})
        plt.ylim(ymin=0)
        #plt.grid()
        #plt.title('Baseline: KL/Distances')
        plt.xlabel('KL', color=fig_color)
        plt.ylabel('Performance increase in relation to baseline', color=fig_color)
        plt.savefig(plotpath + '/correlation_inequality_kl_train_baseline.'+file_type, format=file_type, transparent=True)
        plt.close()

    # Gini
    for i in range(train_steps):
        legend = str(para.c.steps_plastic[i]) + ' training steps, r=' + str(np.round(pearsonr(ginis, np.mean(dists[:,:,i,hpos_idx], axis=0))[0],2))
        plt.errorbar(ginis, np.mean(dists[:,:,i,hpos_idx], axis=0), label=legend, yerr=np.std(dists[:,:,i,hpos_idx], axis=0),  fmt='o',
                     color=train_colors[i], ecolor=np.append(train_colors[i][0:3], 0.5))

    plt.legend(loc=2,prop={'size': legend_size})
    plt.ylim(ymin=0)
    #plt.grid()
    #plt.title('Gini/Distances')
    plt.xlabel('Gini', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/correlation_inequality_gini_train.'+file_type, format=file_type, transparent=True)
    plt.close()