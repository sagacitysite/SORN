from __future__ import division # has to be reimported in every file
#~ import matplotlib # if from ssh
#~ matplotlib.use('Agg')
import ipdb # prettier debugger
import os
import sys
import gc
from importlib import import_module
sys.path.insert(1,"../")

import utils
# This assumes that you are in the folder "common"
utils.initialise_backup(mount="../", dest="../backup")
utils.backup(__file__)

from utils.backup import dest_directory
from common.stats import StatsCollection
from common.sorn import Sorn
from multiprocessing import Pool
import datetime
from common.sources import CountingSource
import cPickle as pickle
import gzip
import numpy as np

# Start debugging mode when an error is raised
def debugger(type,flag):
    print('In debugger! (test_single.py)')
    import ipdb
    ipdb.set_trace()
#~ np.seterrcall(debugger)    
#~ np.seterr(all='call')

def runSORN(c, src):
    # Initialize experiment (source and statictics)
    (source,stats_single,_) = experiment.start(src)

    # Initialize SORN network, has simulation() and step() function
    sorn = Sorn(c,source)

    # Create a StatsCollection and fill it with methods for all statistics
    # that should be tracked (and later plotted)
    stats = StatsCollection(sorn)
    stats.methods = stats_single
    sorn.stats = stats

    # Datalog is used to store all results and parameters
    #stats.dlog.set_handler('*',utils.StoreToH5,
    #                       utils.logfilename("result.h5"))
    #stats.dlog.append('c', utils.unbunchify(c))
    #stats.dlog.set_handler('*',utils.TextPrinter)

    # Final experimental preparations
    experiment.reset(sorn)

    # Start stats
    sorn.stats.start()
    sorn.stats.clear()

    # Run experiment once
    pickle_objects = experiment.run(sorn)

    # Save sources etc
    for key in pickle_objects:
        filename = os.path.join(c.logfilepath,"%s.pickle"%key)
        topickle = pickle_objects[key]
        pickle.dump(topickle, gzip.open(filename,"wb"),
         pickle.HIGHEST_PROTOCOL)

    # Control: Firing-rate model: Substitute spikes by drawing random spikes
    # according to firing rate for each inputindex
    if sorn.c.stats.control_rates:
        experiment.control_rates(sorn)

    # Report stats and close
    stats.single_report()
    stats.disable = True
    #stats.dlog.close()
    sorn.quicksave(filename=os.path.join(c.logfilepath,'net.pickle'))

# Run network for all combinations
def runAll(i):
    j = 0
    # Transitions
    for transitions in transitions_array:

        k = 0
        # Number of excitatory neurons
        for num_neurons in num_neurons_array:

            l = 0
            # Number of input neurons
            for num_input in num_input_array:
                # Set number of excitatory neurons and calculate inhibitory and overall number of neurons
                c.N_e = num_neurons
                c.N_i = int(np.floor(0.2*c.N_e))
                c.N = c.N_e + c.N_i

                # Set number of input neurons
                c.N_u_e = np.floor(num_input*c.N_e)

                # Set transitions and source
                c.source.transitions = transitions
                source = CountingSource(states, transitions, c.N_u_e, c.N_u_i, c.source.avoid)

                # Calculate some more stuff, depending on c.N_e, c.N_i, c.N, etc.
                h_ip_mean = float(2 * c.N_u_e) / float(c.N_e)
                h_ip_range = 0.01
                c.h_ip = np.random.rand(c.N_e) * h_ip_range * 2 + h_ip_mean - h_ip_range

                # Correct connection matrix lamb value
                c.W_ee.lamb = 0.1*c.N_e

                # Print where we are
                print(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ": run " + str(i + 1) + " / model "+
                      str(j + 1) +" / neurons " + str(num_neurons) + " / input " + str(int(np.floor(num_input * c.N_e))))

                # Name of folder for results in this step
                c.multi_name = "run" + str(i) + "_model" + str(j) + "_neurons" + str(k) + "_input" + str(l)
                runSORN(c, source)

                # Free memory
                gc.collect()

                # Increase number of inputs counter
                l += 1

            # Increase number of excitatory counter
            k += 1

        # Increase number of transitions matrices counter
        j += 1


# Parameters are read from the second command line argument
param = import_module(utils.param_file())
experiment_module = import_module(param.c.experiment.module)
experiment_name = param.c.experiment.name
experiment = getattr(experiment_module, experiment_name)(param)

# Initialize parameters
c = param.c

# Store states and remove c.states, otherwise bunch will have a problem
states = c.states
del c.states

# Set logfilepath
c.logfilepath = utils.logfilename('') + '/'

# Set values
num_iterations = range(10)
num_neurons_array = c.N_e
num_input_array = c.N_u_e_coverage
transitions_array = c.source.transitions

total = len(num_iterations) * len(transitions_array) * len(num_neurons_array) * len(num_input_array)

# Stop printing
c.display = False

#for i in num_iterations:
#    runAll(i)

# Start multi processing
pool = Pool(2)
pool.map(runAll, num_iterations)
pool.close()
pool.join()
