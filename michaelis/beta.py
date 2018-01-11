from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import beta
from scipy.stats import uniform

#rcParams.keys()

fig_color = '#000000'
legend_size = 9

rcParams['font.family'] = 'CMU Serif'
rcParams['font.size'] = '12'
rcParams['text.color'] = fig_color
rcParams['axes.edgecolor'] = fig_color
rcParams['xtick.color'] = fig_color
rcParams['ytick.color'] = fig_color
rcParams['grid.color'] = fig_color
rcParams['legend.fancybox'] = True
rcParams['legend.framealpha'] = 0

a, b = 3, 3
x = np.linspace(0, 1, 100)
plt.plot((x-0.5)/5, beta.pdf(x, a, b)/5, color="blue", lw=1)
plt.plot((x-0.5)/5, uniform.pdf(x)/5, color="red", lw=1)
plt.ylim(ymin=0)
plt.savefig('/home/carlo/beta.pdf', dpi=300)
plt.close()
