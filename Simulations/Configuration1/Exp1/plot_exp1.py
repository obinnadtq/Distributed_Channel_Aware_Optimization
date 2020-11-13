import os
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

path_results = os.getcwd()+os.sep+'Exp1'+os.sep+'Results'+os.sep
path_plots = os.getcwd()+os.sep+'Exp1'+os.sep+'Plots'+os.sep

def plot(tikz,datafiles,save_figures):

    key = 'symmetric_3_nodes_sib_101010'

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # plot I(x;z) versus aib iterations after optimizing every node

    # load results
    with open(path_results + key + os.sep + 'Results', 'rb') as f:
        tmp = pickle.load(f)
        print('Results loaded')
        f.close()

    results = tmp.get('results')
    parameters = tmp.get('params')

    fig0 = plt.figure(figsize=(7, 5))
    ax0 = fig0.add_subplot(111)

    last_node = parameters['params'].get('sensor_opt_order')[-1]
    x_values = range(0, results.get('used_iter_aib')+1)
    y_values = np.zeros(results.get('used_iter_aib')+1)
    y_values[0] = results.get('Init_sum_rate')
    for aib_idx in range(0, results.get('used_iter_aib')):
        used_bs_iterations = results['Iterations_aib'][aib_idx]['Node_results'][last_node]['used_iter_bs']
        y_values[aib_idx+1] = results['Iterations_aib'][aib_idx]['Node_results'][last_node]['Iterations_bs'][used_bs_iterations - 1]['I_xz']
    ax0.plot(x_values, y_values, '-+', label='sIB_optimized')

    plt.xticks(np.linspace(0, results.get('used_iter_aib'), results.get('used_iter_aib')+1, dtype=int))
    plt.title('Vector I(x;z) versus iterations of aIB after optimizing every node')
    plt.ylabel('I(x;z)')
    plt.xlabel('Iteration sIB')
    ax0.grid('on')


    if tikz:
        tikz_save(path_plots + "....tex")
    if datafiles:
        np.savetxt(path_plots + '....rst', list(zip([x_values], [y_values])), delimiter='\t')
    if save_figures:
        fig0.savefig(path_plots + '....pdf', bbox_inches='tight', dpi=300, format='pdf')