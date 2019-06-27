from __future__ import division # has to be reimported in every file
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
#from pathos.multiprocessing import ProcessingPool as Pool
import datetime
from common.sources import CountingSource
import cPickle as pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

import copy_reg
import types

# Start debugging mode when an error is raised
def debugger(type,flag):
    print('In debugger! (test_single.py)')
    import ipdb
    ipdb.set_trace()
#~ np.seterrcall(debugger)
#~ np.seterr(all='call')

def runSORN(c, src, testSource):
    # Initialize experiment (source and statictics)
    (source,stats_func) = experiment.start(src, testSource)

    # Initialize SORN network, has simulation() and step() function
    sorn = Sorn(c,source)

    # Create a StatsCollection and fill it with methods for all statistics
    # that should be tracked (and later plotted)
    stats = StatsCollection(sorn)
    stats.methods = stats_func
    sorn.stats = stats

    # Datalog is used to store all results and parameters
    # stats.dlog.set_handler('*',utils.StoreToH5,
    #                        utils.logfilename("result.h5"))
    # stats.dlog.append('c', utils.unbunchify(c))
    # stats.dlog.set_handler('*',utils.TextPrinter)

    # Final experimental preparations
    experiment.reset(sorn)

    # Start stats
    sorn.stats.start()
    sorn.stats.clear()

    # Run experiment once
    pickle_objects = experiment.run(sorn)

    # Backup files
    for key in pickle_objects:
        filename = os.path.join(c.logfilepath,"%s.pickle"%key)
        topickle = pickle_objects[key]
        pickle.dump(topickle, gzip.open(filename,"wb"), pickle.HIGHEST_PROTOCOL)

    # Control: Firing-rate model: Substitute spikes by drawing random spikes
    # according to firing rate for each inputindex
    if sorn.c.stats.control_rates:
        experiment.control_rates(sorn)

    # Report stats and close
    stats.single_report()
    stats.disable = True
    stats.dlog.close()
    sorn.quicksave(filename=os.path.join(c.logfilepath,'net.pickle'))

    # Plot data collected by stats
    #~ dest_directory = os.path.join(dest_directory,'common')
    #experiment.plot_single(dest_directory,
    #                       os.path.join('common','result.h5'))

    # Display figures
    #plt.show()

# Run network for current combinations
def run_all(i):

    j = 0
    # Transitions (Models)
    for transitions in c.source.transitions_array:

        h = 0
        # Hamming thresholds
        for hamming_threshold in c.stats.hamming_threshold_array:

            g = 0
            # Average number of EE-connections
            for connections_density in c.connections_density_array:

                # IP
                #for l in range(np.shape(c.h_ip_array)[1]):  # for h_ip
                l = 0  # for eta_ip / range
                #for eta_ip in c.eta_ip_array:
                for h_ip_range in c.h_ip_range_array:

                    k = 0
                    # Training steps
                    for steps in c.steps_plastic_array:

                        # Set H_IP
                        #c.h_ip = c.h_ip_array[:,l]  # for h_ip
                        #c.eta_ip = eta_ip  # for eta_ip
                        c.h_ip = np.random.rand(c.N_e)*h_ip_range*2 + c.h_ip_mean - h_ip_range  # for h_ip_range

                        # Print where we are
                        #ip_str = str(np.round(np.mean(c.h_ip), 3))  # for h_ip
                        #ip_str = str(np.round(eta_ip, 4))  # for eta_ip
                        ip_str = str(h_ip_range)  # for h_ip_range

                        print(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") +": run "+ str(i+1) +" / model "+ str(j+1) +" / threshold "+ str(hamming_threshold) +" / ip "+ ip_str +" / connections "+ str(connections_density*c.N_e) +" / "+ str(steps))

                        testSource = None
                        if c.source.testing:
                            # If testing input is given, train network with given training matrix
                            c.source.transitions = c.source.training
                            source = CountingSource(states, c.source.training, c.N_u_e, c.N_u_i, c.source.avoid)
                            # If testing is varied, add another Counting Source for testing phase, which is given by current loop
                            testSource = CountingSource(states, transitions, c.N_u_e, c.N_u_i, c.source.avoid)
                        else:
                            # If input is only given while training, vary transitions while training (as usual)
                            c.source.transitions = transitions
                            source = CountingSource(states, transitions, c.N_u_e, c.N_u_i, c.source.avoid)

                        # Set steps_plastic and correct N_steps
                        c.steps_plastic = steps
                        c.N_steps = c.steps_plastic + c.steps_noplastic_train + c.steps_noplastic_test

                        # Set hamming threshold
                        c.stats.hamming_threshold = hamming_threshold

                        # Set number of average EE-connections
                        c.W_ee.lamb = connections_density*c.N_e

                        # Name of folder for results in this step
                        c.file_name = "run"+str(i)
                        c.state.index = (j, k, h, l, g)  # models, training steps, threshold, h_ip, #EE-connections
                        runSORN(c, source, testSource)

                        # Free memory
                        gc.collect()

                        # Increase trainig steps counter
                        k += 1

                    # H_IP counter needs no increase (is range) (for h_ip)
                    l+=1 # for eta_ip

                # Increase average # EE-connections counter
                g += 1

            # Increase hamming threshold counter
            h += 1

        # Increase models counter
        j += 1

# Parameters are read from the second command line argument
param = import_module(utils.param_file())
experiment_module = import_module(param.c.experiment.module)
experiment_name = param.c.experiment.name
experiment = getattr(experiment_module, experiment_name)(param)

# Initialize parameters
c = param.c
c.state = utils.Bunch()

# Store states and remove c.states, otherwise bunch will have a problem
states = c.states
del c.states

# Set logfilepath
c.logfilepath = utils.logfilename('') + '/'
c.steps_plastic_array = c.steps_plastic
c.source.transitions_array = c.source.transitions
c.stats.hamming_threshold_array = c.stats.hamming_threshold
#c.h_ip_array = c.h_ip
#c.eta_ip_array = c.eta_ip
c.h_ip_range_array = c.h_ip_range
c.connections_density_array = c.connections_density

# Set values
num_iterations = range(c.N_iterations)

## Start multi processing
#pool = Pool(3)
#pool.map(run_all, num_iterations)
#pool.close()
#pool.join()

for i in num_iterations:
	run_all(i)
#run_all(0)
#run_all(1)
