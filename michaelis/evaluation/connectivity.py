from evaluation import *

def plot_performance(distances, ymaxi=0.4):
    print('# Connectivity: Performance plot')
    # distances: runs, models, training, connectivity, test chunks
    last_chunk = np.shape(distances)[4]-1
    num_con = np.shape(distances)[3]
    last_train = np.shape(distances)[2]-1
    num_models = np.shape(distances)[1]
    
    static = np.mean(distances[:,:,0,:,last_chunk], axis=0)
    static_std = np.std(distances[:,:,0,:,last_chunk], axis=0)
    plastic = np.mean(distances[:,:,last_train,:,last_chunk], axis=0)
    plastic_std = np.std(distances[:,:,last_train,:,last_chunk], axis=0)
    # models, connectivity
    
    cols = cm.rainbow(np.linspace(0, 1, num_con))
    
    # Plastic
    for i in range(num_con):
        legend = r"$\rho$ = " + str(para.c.connections_density[i])
        plt.errorbar(np.arange(num_models)+1, plastic[:,i], label=legend, yerr=plastic_std[:,i], marker='o',
                     color=cols[i], ecolor=np.append(cols[i][0:3], 0.5))

    plt.legend(prop={'size': legend_size})
    plt.xlim(xmin=0.5, xmax=num_models+0.5)
    plt.ylim(ymin=0,ymax=ymaxi)
    plt.xlabel('Model', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/connectivity_plastic.'+file_type, format=file_type, transparent=True)
    plt.close()
    
    # Static
    for i in range(num_con):
        legend = r"$\rho$ = " + str(para.c.connections_density[i])
        plt.errorbar(np.arange(num_models)+1, static[:,i], label=legend, yerr=static_std[:,i], marker='o',
                     color=cols[i], ecolor=np.append(cols[i][0:3], 0.5))

    plt.legend(prop={'size': legend_size})
    plt.xlim(xmin=0.5, xmax=num_models+0.5)
    plt.ylim(ymin=0,ymax=ymaxi)
    plt.xlabel('Model', color=fig_color)
    plt.ylabel('Transition error', color=fig_color)
    plt.savefig(plotpath + '/connectivity_static.'+file_type, format=file_type, transparent=True)
    plt.close()
