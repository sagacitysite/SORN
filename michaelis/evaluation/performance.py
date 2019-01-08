import evaluation as ev
from evaluation import *


def manual_plots():
    print('# Manual performance plot')

    # 3stat 4stat1 4stat2 5stat
    data = [
        { 'title': 'L1 error - Spontaneous activity', 'ylabel': 'L1 error', 'err': [0.160, 0.038, 0.024, 0.120]},
        { 'title': 'L2 error - Spontaneous activity', 'ylabel': 'L2 error', 'err': [0.185, 0.056, 0.034, 0.137]},
        { 'title': 'L1 error - weight clusters', 'ylabel': 'L1 error', 'err': [0.060, 0.069, 0.097, 0.079]},
        { 'title': 'L2 error - weight clusters', 'ylabel': 'L2 error', 'err': [0.069, 0.077, 0.114, 0.083]},
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

