from evaluation import *

def plot(normed_stationaries):
    print('Lorenz: Plot')
    #global plotpath, para
    # runs, models, train steps, test chunks, stationary

    # Calculate stationary distributions from markov chains
    stationaries = np.array([stationaryDistribution.calculate(transition) for transition in para.c.source.transitions])

    # Preparation
    n = np.shape(stationaries)[1]
    num_models = np.shape(stationaries)[0]
    x = np.repeat(1 / float(n), n)

    # Define color palette
    color_palette = cm.rainbow(np.linspace(0, 1, num_models))

    x0 = np.append(0, np.cumsum(x))
    plt.plot(x0, x0, color='black')

    for i in range(num_models):
        y = np.sort(stationaries[i])
        y0 = np.append(0, np.cumsum(y))
        plt.plot(x0, y0, label= 'Model'+str(i+1), color=color_palette[i])

    plt.legend(loc=2, prop={'size': legend_size})
    plt.ylim(ymax=1)
    #plt.grid()
    #plt.title('Lorenz curve: Stationary distributions')
    plt.xlabel('States', color=fig_color)
    plt.ylabel('Cumulative probability', color=fig_color)
    plt.savefig(plotpath + '/lorenz-curve.'+file_type, format=file_type, transparent=True)
    plt.close()
