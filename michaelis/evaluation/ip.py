from evaluation import *

def plot(distances, legend_labels=None):
    print('Intrinsic plasticity: Plot')
    
    # distances: runs, models, h_ip, test chunks
    last_chunk = np.shape(distances)[3]-1
    #num_hip = np.shape(distances)[2]
    num_models = np.shape(distances)[1]
    
    dists_mean = np.mean(distances[:,:,:,last_chunk], axis=0)
    dists_std = np.std(distances[:,:,:,last_chunk], axis=0)
    # dists: models, h_ip
    
    num_hip = np.shape(dists_mean)[1]
    
    cols = cm.rainbow(np.linspace(0, 1, num_hip))
    
    lab = None
    if legend_labels is None:
        lab = '$c_{IP}$'
    else:
        lab = legend_labels
    
    for i in range(num_hip):
        legend = str(lab + ' = ' + str(PARA.c[IP][i]))
        plt.errorbar(np.arange(num_models)+1, dists_mean[:,i], label=legend, yerr=dists_std[:,i], marker='o', # fmt='o',
                     color=cols[i], ecolor=np.append(cols[i][0:3], 0.5))

    plt.legend(prop={'size': LEGEND_SIZE})
    plt.xlim(xmin=0.5, xmax=num_models+0.5)
    plt.ylim(ymin=0)
    plt.xlabel('Model', color=FIG_COLOR)
    plt.ylabel('Transition error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/h_ip.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
    
    
    plt.errorbar(PARA.c[ip], dists_mean[0,:], yerr=np.std(distances[:,0,:,last_chunk], axis=0), fmt='o')
    plt.xlim(xmin=np.min(PARA.c[ip])-0.5, xmax=np.max(PARA.c[ip])+0.5)
    plt.ylim(ymin=0)
    plt.xlabel('IP factor', color=FIG_COLOR)
    plt.ylabel('Transition error', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/h_ip_model1.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
    
    # t-Tests
    file = ""
    for i in range(num_hip):
        for j in range(num_hip):
            if j > i:
                res =scipy.stats.ttest_ind(distances[:,0,i,last_chunk], distances[:,0,j,last_chunk])
                file += "c_IP "+str(PARA.c[ip][i])+" vs. "+str(PARA.c[ip][j])+": t="+str(res[0])+", p="+str(res[1])+"\n"
    text_file = open(PLOTPATH + "/t-tests_h_ip_factor.txt", "w")
    text_file.write(file)
    text_file.close()

def plot_activity(activity):
    print('Intrinsic plasticity: Plot activity')
    
    # activity: runs, models, h_ip, test chunks
    last_chunk = np.shape(activity)[3]-1
    #num_hip = np.shape(activity)[2]
    num_models = np.shape(activity)[1]
    
    act_mean = np.mean(activity[:,0,:,last_chunk], axis=0)
    act_std = np.std(activity[:,0,:,last_chunk], axis=0)
    # act: h_ip
    
    num_hip = np.shape(act_mean)[0]
    
    plt.bar(PARA.c[ip]-0.125, act_mean, 0.25, linewidth=0, yerr=act_std)
    plt.xlim(xmin=np.min(PARA.c[ip])-0.5, xmax=np.max(PARA.c[ip])+0.5)
    plt.ylim(ymin=0)
    plt.xlabel('IP factor', color=FIG_COLOR)
    plt.ylabel('Average activity', color=FIG_COLOR)
    plt.savefig(PLOTPATH + '/h_ip_activity_model1.'+FILE_TYPE, format=FILE_TYPE, transparent=True)
    plt.close()
