from __future__ import division
import numpy as np
import utils
utils.backup(__file__)

# see this file for parameter descriptions
from common.defaults import *

c.N_iterations = 5  # 20

c.N_e = 200  #200 # TODO if this is changed, eta_stdp has also to change ??
c.N_i = int(np.floor(0.2*c.N_e))
c.N = c.N_e + c.N_i
c.N_u_e = np.floor(0.05*c.N_e) #np.floor(0.05*c.N_e) # 10 connections from any input to the excitatory neurons
c.N_u_i = 0

c.double_synapses = False

# used Bunch (https://pypi.python.org/pypi/bunch)
c.connections_density = np.array([0.1])  # np.array([0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4])  # default: np.array([0.1])
c.W_ee = utils.Bunch(use_sparse=True,
                     lamb=c.connections_density*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp = 0.001,
                     sp_prob = 0.0,
                     sp_initial=0.000,
                     no_prune = True,
                     upper_bound = 1,
                     weighted_stdp = False,
                     eta_ds = 0.1
                     )

c.W_ei = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False,
                     eta_istdp = 0.0,
                     h_ip=0.1)

c.W_ie = utils.Bunch(use_sparse=False,
                     lamb=np.inf,
                     avoid_self_connections=False)

c.steps_plastic = np.array([50000])  #np.array([0, 2500, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]) # default: np.array([50000])
c.steps_noplastic_train = 50000  #50000
c.steps_noplastic_test = 40000  #40000
c.display = False  # False: Stop displaying stuff

# 0.1 -> ~30% noisespikes, 0.05 -> ~15%, 0.01 -> ~2.5%, 0.005 -> ~1%
c.noise_sig = 0 #0.045

c.with_plasticity = True

c.input_gain = 0.5

#c.eta_ip = np.array([0.0002, 0.001, 0.002]) # np.arange(0.0002,0.0021,0.0004) #np.arange(0.0002,0.0021,0.0002) # Default: np.array([0.001])
c.eta_ip = 0.001

c.h_ip_mean = float(2*c.N_u_e)/float(c.N_e)
c.h_ip_range = np.array([0.01])  #np.array([0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])  # np.array([0.01])
#h_ip_range = 0.01

#c.h_ip = np.random.rand(c.N_e)*h_ip_range*2 + h_ip_mean - h_ip_range
#c.h_ip_factor = np.arange(0.5,3.51,0.5) # Default: np.array([2])
#h_ip_mean = c.h_ip_factor*float(c.N_u_e)/float(c.N_e)
#c.h_ip = np.broadcast_to(np.random.rand(c.N_e), (len(c.h_ip_factor),200)).T*h_ip_range*2 + np.broadcast_to(h_ip_mean, (200,len(c.h_ip_factor))) - h_ip_range

c.always_ip = True
c.synaptic_scaling = True

c.T_e_max = 0.5
c.T_e_min = 0.0
c.T_i_max = 0.35
c.T_i_min = 0.0
c.ordered_thresholds = True

c.fast_inhibit = True
c.k_winner_take_all = False
c.ff_inhibition = False
c.ff_inhibition_broad = 0.0

c.experiment.module = 'michaelis.experiment_mcmc'  # michaelis.experiment_mcmc_withSTDP
c.experiment.name = 'Experiment_mcmc'  # experiment_mcmc_withSTDP

#######################################
c.stats.file_suffix = 'MCMC_ds_noSTDP'
#######################################
c.stats.save_spikes = True
c.stats.quenching = 'train'
c.stats.quenching_window = 2
c.stats.match = False
c.stats.lstsq_mue = 1
c.stats.control_rates = False
c.stats.ISI_step = 4
c.stats.transition_step_size = 5000  # 5000
c.stats.ncomparison_per_state = 500  # If ncomparison or only_last should be used, remove this parameter
c.stats.hamming_threshold = np.array([np.inf]) #np.append(np.arange(20,27,2), np.inf)  #np.append(np.arange(20,41,5), np.inf)  #np.append(np.arange(12,23,2), np.inf) # Use not threshold: np.array([np.inf])
# c.stats.only_last = 3000  # affects many stats: take only last x steps

# Following parameters for randsource
c.source.use_randsource = False
#c.source.N_words = 10
#c.source.N_letters = 10
#c.source.word_length = np.array([2,6])
#c.source.max_fold_prob = 3

c.source.prob = 0.75 # This is only here to be changed by cluster
c.source.avoid = False
c.source.control = False # For sequence_test

# For cluster: same randsource for entire simulation
#~ from common.sources import CountingSource
#~ source = CountingSource.init_simple(
        #~ c.source.N_words,
        #~ c.source.N_letters,c.source.word_length,
        #~ c.source.max_fold_prob,c.N_u_e,c.N_u_i,
        #~ c.source.avoid, seed=42)
#~ import random
#~ print source.words, np.random.randint(0,500), random.random()

#~ from common.sources import RandomLetterSource
#~ source = RandomLetterSource(c.source.N_letters,c.N_u_e,c.N_u_i,
                            #~ c.source.avoid)
from common.sources import CountingSource
c.states = ['A','B','C','D']

#c.source.transitions = np.array([[[0, 1],
#                                  [1, 0]]])

c.source.testing = True  # If False: apply Markov chain variation while learning, if True: apply while testing

# Only important if c.source.testing is True
#c.source.training = np.array([[0.01, 0.97, 0.01, 0.01],
#                              [0.01, 0.01, 0.97, 0.01],
#                              [0.01, 0.01, 0.01, 0.97],
#                              [0.97, 0.01, 0.01, 0.01]])

# Models 7 = VII
transitions = []
iterate = np.array([0., 0.0005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.23, 0.24, 0.25])
#iterate = np.array([0., 0.1, 0.25])
for it in iterate:
#    transitions.append([[it, 1.-(2*it), it],
#                        [it, it, 1.-(2*it)],
#                        [1.-(2*it), it, it]])
    transitions.append([[it, 1.-(3*it), it, it],
                        [it, it, 1.-(3*it), it],
                        [it, it, it, 1.-(3*it)],
                        [1.-(3*it), it, it, it]])
#    transitions.append([[it, 1.-(4*it), it, it, it],
#                        [it, it, 1.-(4*it), it, it],
#                        [it, it, it, 1.-(4*it), it],
#                        [it, it, it, it, 1.-(4*it)],
#                        [1.-(4*it), it, it, it, it]])
c.source.transitions = np.array(transitions)

c.wait_min_plastic = 0
c.wait_var_plastic = 0
c.wait_min_train = 0
c.wait_var_train = 0

# Cluster
#c.cluster.vary_param = 'steps_plastic'#'with_plasticity'#
#c.cluster.params = np.linspace(5000,15000,3)#[False,True]#
#if c.imported_mpi:
#    c.cluster.NUMBER_OF_SIMS  = len(c.cluster.params)
#    c.cluster.NUMBER_OF_CORES = MPI.COMM_WORLD.size
#    c.cluster.NUMBER_LOCAL = c.cluster.NUMBER_OF_SIMS // c.cluster.NUMBER_OF_CORES

def test():
    pass
