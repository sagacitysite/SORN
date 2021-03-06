import numpy as np
import os
import sys
import types
import scipy
from scipy.stats import pearsonr

"""
Import matplotlib
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

"""
Prepare path constants
"""

# Get backup folder from argument
os.chdir("../backup/test_multi/" + str(sys.argv[1]))  #"2017-12-31_12-01-55_models5"

# Get current working dir and append to sys path
PATH = os.getcwd()
sys.path.append(PATH)

# Define/Create specific paths
DATAPATH = PATH + "/data"
PLOTPATH = PATH + "/plots"
if not os.path.exists(PLOTPATH):
    os.mkdir(PLOTPATH)

"""
Initialize evaluation module
"""

# Load parameter
from . import _parameter
PARA, NUM_RUNS, IP, NUM_CHUNKS, FIG_COLOR, LEGEND_SIZE, FILE_TYPE = _parameter.load()

# Import all other evaluation functions
from . import _data
from . import _helper
from . import activity
from . import connectivity
from . import entropy
from . import hamming
from . import ip
from . import lorenz
from . import performance
from . import performance_correlation
from . import spikes
from . import training
from . import transition_matrix
from . import test_trace
from . import weights
