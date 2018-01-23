from __future__ import division
from pylab import *
import utils
utils.backup(__file__)

from michaelis.plot_single import plot_results as plot_results_single
from michaelis.plot_cluster import (plot_results
                                    as plot_results_cluster)
from common.sources import CountingSource, TrialSource, NoSource
from common.experiments import AbstractExperiment
from common.sorn_stats import *

class Experiment_mcmc_withSTDP(AbstractExperiment):
    def start(self, src):
        super(Experiment_mcmc_withSTDP,self).start()
        c = self.params.c

        if not c.has_key('display'):
            c.showPrint = True

        # In case we are using cluster calculation?
        if self.cluster_param == 'source.prob':
            prob = c.source.prob
            assert(prob>=0 and prob<=1)
            self.params.source = CountingSource(
                                          self.params.source.words,
                                          np.array([[prob,1.0-prob],
                                                    [prob,1.0-prob]]),
                                          c.N_u_e,c.source.avoid)

        # Random source? Parameter is false, so actually not used
        if c.source.use_randsource:
            self.params.source = CountingSource.init_simple(
                    c.source.N_words,
                    c.source.N_letters,c.source.word_length,
                    c.source.max_fold_prob,c.N_u_e,c.N_u_i,
                    c.source.avoid)

        # Make inputsource out of source by using TrialSource object from sources.py
        # It now has some nice methods to deal with the network, like generate connections, etc.
        self.inputsource = TrialSource(src,
                         c.wait_min_plastic, c.wait_var_plastic, 
                         zeros(src.N_a), 'reset')
        
        # Stats
        inputtrainsteps = c.steps_plastic + c.steps_noplastic_train  # steps during self-organization + steps for first phase w/o plasticity

        # Begin: For PatternProbabilityStat
        # Set burn-in phase to half of steps for 2nd phase w/o plasticity, but maximum 3000
        if c.steps_noplastic_test > 6000:
            burnin = 3000
        else:
            burnin = c.steps_noplastic_test//2

        # Shuffle indices of excitatory neurons
        shuffled_indices = arange(c.N_e)
        np.random.shuffle(shuffled_indices)

        N_subset = 8 # 8
        start_train = c.steps_plastic+burnin  # step when training begins
        half_train = start_train+(inputtrainsteps-start_train)//2  # step when network is half trained
        start_test = inputtrainsteps+burnin  # step when testing begins
        half_test = start_test+(c.N_steps-start_test)//2  # step when network is half tested
        # End: For PatternProbabilityStat

        # Initialize statistics
        stats_all = [
                     InputIndexStat(),
                     SpikesStat(),
                     InputUnitsStat(),
                     NormLastStat(),
                     SpontPatternStat(),
                     ParamTrackerStat(),
                     EvokedPredStat(
                            traintimes=[c.steps_plastic,
                                        c.steps_plastic+
                                        c.steps_noplastic_train//2],
                            testtimes =[c.steps_plastic+
                                        c.steps_noplastic_train//2,
                                        c.steps_plastic+
                                        c.steps_noplastic_train],
                                        traintest=c.stats.quenching),
                    ]

        stats_single = [
                         ActivityStat(),
                         #MeanActivityStat(),
                         # SpikesStat(inhibitory=True),
                         # ISIsStat(interval=[start_test,c.N_steps]),
                         # ConnectionFractionStat(),
                         # EndWeightStat(),
                         # #~ BalancedStat(), # takes lots of time and mem
                         # CondProbStat(),
                         # SpontIndexStat(),
                         # SVDStat(),
                         # SVDStat_U(),
                         # SVDStat_V(),
                         SpontTransitionAllStat(),
                         SpontTransitionStat(),
                         SpontTransitionDistance(),
                         EEWeightStats()
                         # InputUnitsStat(),
                         # PatternProbabilityStat(
                         #                [[start_train,half_train],
                         #                 [half_train,inputtrainsteps],
                         #                 [start_test,half_test],
                         #                 [half_test,c.N_steps]],
                         #                  shuffled_indices[:N_subset]),
                         # WeightHistoryStat('W_ee',record_every_nth=100),
                         # WeightHistoryStat('W_eu',
                         #                   record_every_nth=9999999)
                        ]

        if c.double_synapses:  # if we have two excitatory synapses for each connection
            stats_single += [WeightHistoryStat('W_ee_2',
                                           record_every_nth=100)]

        # Return inputsource and statistics
        return (self.inputsource,stats_all+stats_single,stats_all)
        
    def reset(self,sorn):
        super(Experiment_mcmc_withSTDP,self).reset(sorn)
        c = self.params.c
        stats = sorn.stats # init sets sorn.stats to None
        sorn.__init__(c,self.inputsource)
        sorn.stats = stats
            
    def run(self,sorn):
        super(Experiment_mcmc_withSTDP,self).run(sorn)
        c = self.params.c

        # If filename was not set, we are in single mode
        if not c.has_key('file_name'):
            c.file_name = "single"

        # If state does not exist, create it
        if not c.has_key('state'):
            c.state = utils.Bunch()

        if c.display == True:
            print('Run self organization:')
        sorn.simulation(c.steps_plastic)

        # RUN SELF ORGANISATION
        if c.display == True:
            print('Run self organization:')
        sorn.simulation(c.steps_plastic)

        # RUN TRAINING
        if c.display == True:
            print('\nRun training:')

        # Stop STDP and synaptic normalization
        sorn.update = False

        # Run with trials
        # self.inputsource.source.N_a: Size of input source (letters, sequences), e.g. 4
        trialsource = TrialSource(self.inputsource.source, 
                                  c.wait_min_train, c.wait_var_train, 
                                  zeros(self.inputsource.source.N_a), 
                                  'reset')

        sorn.source = trialsource
        shuffle(sorn.x) # {0,1}
        shuffle(sorn.y) # {0,1}

        sorn.simulation(c.steps_noplastic_train)

        # RUN TESTING
        if c.display == True:
            print('\nRun testing:')

        # Start STDP and synaptic normalization again
        sorn.update = True

        # Run with spont (input u is always zero)
        spontsource = NoSource(sorn.source.source.N_a)
        sorn.source = spontsource
        shuffle(sorn.x)
        shuffle(sorn.y)
        # Simulate spontaneous activity
        sorn.c.ff_inhibition_broad = 0
        if not c.always_ip:
            sorn.c.eta_ip = 0

        sorn.simulation(c.steps_noplastic_test)
        
        return {'source_plastic':self.inputsource,
                'source_train':trialsource,
                'source_test':spontsource}
     
    def plot_single(self,path,filename):
        #plot_results_single(path,filename,self.params.c)
        pass
    def plot_cluster(self,path,filename):
        #plot_results_cluster(path,filename)
        pass
