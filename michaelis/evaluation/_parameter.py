from evaluation import *

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
    if(num_runs == None):
        num_runs = para.c.N_iterations
    
    # Evaluate ip parameter from parameter file
    ip_param = None
    if('h_ip_factor' in para.c):
        ip_param = 'h_ip_factor'
    else:
        if(np.isscalar(para.c.h_ip_range)):
            ip_param = 'eta_ip'
        elif(np.isscalar(para.c.eta_ip)):
            ip_param = 'h_ip_range'
            
    return para, num_runs, ip_param
