from evaluation import *

"""
@desc: Configure plots
"""
def plots():
    print('# Configure plots')
    
    fig_color = '#000000'
    legend_size = 12  # 12/14
    file_type = 'png'  # svg, pdf

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
@desc: Configure data sources
"""
def sources():
    print('# Configure sources')
    
    sources = { 'transition_distances': None, 'activity': None, 'estimated_stationaries': None, 'ncomparison': None, 'hamming_distances': None, 'weights_ee': None, 'weights_eu': None, 'norm_last_input_spikes': None, 'norm_last_input_index': None }
    #sources = { 'transition_distances': None, 'activity': None, 'estimated_stationaries': None, 'ncomparison': None, 'norm_last_input_spikes': None, 'norm_last_input_index': None }
    return sources

