from evaluation import *

"""
@desc: Configure plots
"""
def get_plot_parameters():
    fig_color = '#000000'
    legend_size = 12  # 12/14
    file_type = 'svg'  # png, svg, pdf

    rcParams['font.family'] = 'CMU Serif'
    rcParams['font.size'] = '14'  # 14/20
    rcParams['text.color'] = fig_color
    rcParams['axes.edgecolor'] = fig_color
    rcParams['xtick.color'] = fig_color
    rcParams['ytick.color'] = fig_color
    rcParams['axes.grid'] = True
    rcParams['grid.linestyle'] = ':'
    rcParams['grid.linewidth'] = 0.5
    rcParams['grid.color'] = fig_color
    rcParams['legend.fancybox'] = True
    rcParams['legend.framealpha'] = 0.2  # 0.75
    rcParams['patch.linewidth'] = 0
    #rcParams['figure.autolayout'] = True
    
    #print(rcParams.keys())
    
    return fig_color, legend_size, file_type


"""
@desc: Load parameters
"""
def load(parafile = 'param_mcmc_multi', num_runs=None):
    print('# Load parameters')
    
    # Import utils and imp
    import utils
    import imp
    
    # Overwrite utils backup function to avoid new backup copies
    def backup_overwrite(a, b=None):
        pass
    utils.backup = backup_overwrite
    
    # Load param file and return
    para = imp.load_source('param_mcmc_multi', PATH+'/michaelis/'+parafile+'.py')
    
    # Set number of runs to evaluate to parameter file if no parameter was set
    # or chosen num_runs are bigger than avalaible data
    if num_runs == None or num_runs > para.c.N_iterations:
        num_runs = para.c.N_iterations
    
    # Evaluate ip parameter from parameter file
    ip_param = None
    if 'h_ip_factor' in para.c:
        ip_param = 'h_ip_factor'
    else:
        if(np.isscalar(para.c.h_ip_range)):
            ip_param = 'eta_ip'
        elif(np.isscalar(para.c.eta_ip)):
            ip_param = 'h_ip_range'

    # Calculate number of chunks
    chunk_size = para.c.stats.transition_step_size
    spont_spikes_steps = para.c.steps_noplastic_test
    num_chunks = int(round((spont_spikes_steps / chunk_size) - 0.5))

    # Load plot parameters
    fig_color, legend_size, file_type = get_plot_parameters()
            
    return para, num_runs, ip_param, num_chunks, fig_color, legend_size, file_type
