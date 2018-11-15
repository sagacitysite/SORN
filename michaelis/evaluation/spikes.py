from evaluation import *
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def plot_train_histograms(last_input_spikes):
    print('# Spikes: Train histogram')
    num_models = np.shape(last_input_spikes)[0]
    num_states = len(para.c.states)
    num_neurons = para.c.N_e
    
    ncom = para.c.stats.ncomparison_per_state
    state_names = para.c.states
    
    for i in range(num_models):
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        fig.suptitle('Model: '+str(i+1))
        
        for j in range(num_states):
            # Mean data, data was already sorted before saving
            state_mean = np.mean(last_input_spikes[i,:,ncom*j:ncom*(j+1)], axis=1)

            # Cross validate bandwith of kernel estimation
            params = {'bandwidth': np.logspace(-5, 0.05, 50)}
            grid = GridSearchCV(KernelDensity(kernel='gaussian'), params)
            grid.fit(state_mean[:,None])

            # User best model to estimate new values for plot
            kde = grid.best_estimator_
            xvals = np.linspace(0,1,100)
            estimates = np.exp(kde.score_samples(xvals[:,None]))

            # Plot histogram and density
            ax = plt.subplot(gs[j])
            ax.hist(state_mean, bins=50, range=(0,1), color='#F44336', alpha=0.9)
            ax.plot(xvals, estimates, color='#3F51B5')
            ax.set_title('State: '+state_names[j])
            ax.set_xlim(xmin=0, xmax=1)
            ax.set_ylim(ymin=0, ymax=120)
		
        plt.savefig(plotpath + '/training_activity_histogram_model'+ str(i+1) +'.'+file_type, format=file_type, transparent=True)
        plt.close()
