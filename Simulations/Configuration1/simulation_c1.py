import sys
import os
import numpy as np
import pickle
import logging
from itertools import permutations

#check if script was called by multiprocessing script
if len(sys.argv) > 3:
    multiprocessing = True
else:
    # setup all path variables and logging
    multiprocessing = False
    sys.path.append(os.path.abspath('..'))
    import py_setup
    py_setup

from sequential_distributed_IB_class import SequentialDistributedIB

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

# Configuration 1 (explanation of configuration1)
#   exp1: explanation of exp1

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------


#helper functions
#-------------------------------------------------------------------------------------------------------
def put_in_position(x_, pos_, val_):
    newx = np.copy(x_)
    np.put(newx, pos_, val_)
    return newx

def increase_dims(x_,dim_tuple_):
    newx = np.copy(x_)
    for i in range(0,len(dim_tuple_)):
        newx = np.expand_dims(newx,dim_tuple_[i])
    return newx

def parseParameter(key,path_exp):
    params = {}
    lines = [line.rstrip('\n') for line in open(path_exp+key+'.exp')]
    for line in lines:
        if line != '':
            line = line.strip()
            line = line.replace(' ','')
            tmp = line.split('=')
            if len(tmp[1].split(',')) > 1:
                list_tmp = tmp[1].replace('[', '').replace(']', '')
                list_tmp = list_tmp.split(',')
                idx = 0
                for val in list_tmp:
                    if is_int(val):
                        list_tmp[idx] = int(val)
                    elif is_float(val):
                        list_tmp[idx] = float(val)
                    elif val == 'inf':
                        list_tmp[idx] = np.inf
                    idx+=1
                # globals().update({tmp[0]: list_tmp})
                params[tmp[0]] = list_tmp
            elif tmp[1][0] == '[':
                val = tmp[1].replace('[','').replace(']','')
                if is_int(val):
                    val = [int(val)]
                elif is_float(val):
                    val = [float(val)]
                # globals().update({tmp[0]: val})
                params[tmp[0]] = val
            elif is_int(tmp[1]):
                val = int(tmp[1])
                # globals().update({tmp[0]: val})
                params[tmp[0]] = val
            elif is_float(tmp[1]):
                val = float(tmp[1])
                # globals().update({tmp[0]: val})
                params[tmp[0]] = val
            elif tmp[1] == 'None':
                val = None
                # globals().update({tmp[0]: val})
                params[tmp[0]] = val
            elif tmp[1]== 'True' or tmp[1] == 'False':
                str_val = tmp[1].replace('\'', '')
                val = eval(str_val)
                # globals().update({tmp[0]: val})
                params[tmp[0]] = val
            else:
                str_val = tmp[1].replace('\'','')
                # globals().update({tmp[0]:str_val})
                params[tmp[0]] = str_val
    return params

def is_int(value):
    try:
        int(value)
        return True
    except:
        return False

def is_float(value):
    try:
        float(value)
        return True
    except:
        return False

def create_init_quantizer(type, Ny , Nz, Nx, N_sensors, p_xyi = None):
    logging.debug('create initial quantizers')

    if type == 'uniform':
        # create uniform quantizers
        p_init = [[] for _ in range(0, N_sensors)]
        for i in range(0, N_sensors):
            N_ones = np.floor(Ny[i] / Nz[i])
            p_tmp = np.zeros([Nz[i], Ny[i]])
            for run in range(0, Nz[i]):
                ptr = range(run * int(N_ones), min(((run + 1) * int(N_ones)), Ny[i]))
                p_tmp[run, ptr] = 1
            if ptr[-1] < Ny[i]:
                p_tmp[run, range(ptr.stop, np.size(p_tmp, 1))] = 1
            p_init[i] = p_tmp
        return p_init, None

    if type == 'max_entropy':
        p_yi = [[] for _ in range(0, N_sensors)]
        for i in range(0, N_sensors):
            p_yi[i] = np.sum(p_xyi[i],0)

        # Maximum Output Entropy (MOE) initialization
        p_init = [[] for _ in range(0, N_sensors)]
        for i in range(0, N_sensors):
            p_tmp = np.zeros([Nz[i], Ny[i]])
            runy = 0
            start = 0
            for runz in range(0,Nz[i]):
                condition = 1
                while condition:
                    p_tmp[runz,runy] = 1
                    runy+=1
                    if runy >= Ny[i]:
                        runy -=1
                        break
                    with np.errstate(all='ignore'):
                        condition = np.sum(p_yi[i][start:runy])<= (1-np.sum(p_yi[i][0:runy-1]))/((Nz[i]-1)-runz)
                start = runy
            p_init[i] = p_tmp
        return p_init, None


def create_alphabet(type):
    if type == '4-ASK':
        Nx = 4
        alphabet = np.array([-3, -1, 1, 3])  # for comparism -3,-1,1,3
        p_x = [1 / Nx for _ in alphabet]
    return alphabet, p_x

def create_channel_distribution(alphabet, p_x, SNR_db_sensors, N_sensors, Nx ,Ny):
    # joint dist p_xy
    # power of AWGN at sensor nodes
    SNR_lin = [np.power(10, (x / 10)) for x in SNR_db_sensors]
    sigma2_N = np.tile(np.dot(np.power(alphabet, 2), p_x), (1, N_sensors)) / SNR_lin

    # seperate p(x,yi) for every sensor
    p_xyi = [[] for _ in range(0, N_sensors)]
    for s_idx in range(0,N_sensors):
        dy = 2 * (np.max(alphabet) + 5 * np.sqrt(sigma2_N[0, s_idx])) / Ny[s_idx]
        y = np.around(np.arange(-(np.max(alphabet) + 5 * np.sqrt(sigma2_N[0, s_idx])), np.max(alphabet) + 5 * np.sqrt(sigma2_N[0, s_idx]), dy), 4)
        p_xyi[s_idx] = np.ones((Nx,Ny[s_idx]))
        for x_idx in range(0,Nx):
            p_xyi[s_idx][x_idx,:] = np.exp(-np.power((y[np.arange(0, Ny[s_idx])] - alphabet[x_idx]), 2) / 2 / sigma2_N[0, s_idx]) * p_x[x_idx]
        p_xyi[s_idx] = p_xyi[s_idx] / np.sum(p_xyi[s_idx])

    return p_xyi

def result_check(path_results, key, name):
    if os.path.exists(path_results + key + os.sep + name):
        return True
    else:
        return False

#-------------------------------------------------------------------------------------------------------------------
#---- main functionality -------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

def main(exp = None, key = None):

    # system flags
    load = 0  # load == 1 -> loads results of simulation with key -> to check results
    if(len(sys.argv)>1):
        console = 1 #-> script runs on extern console
    else:
        console = 0


    if console: # if script runs from extern -> usually simulation is done
        load = 0

    if console:
        if not multiprocessing:
            exp = sys.argv[1]
            key = sys.argv[2]
    else:
        exp = 'Exp1'

        key = 'symmetric_3_nodes_sib_101010'

    # path variables
    path_results = os.getcwd()+os.sep+exp+os.sep+'Results'+os.sep
    path_exp = os.getcwd()+os.sep+exp+os.sep+'Exp_files'+os.sep

    if multiprocessing:
        exp = exp.split(os.sep)[1]

    if load:
        load_key = ''
    #-------------------------------------------------------------------------------------------------------

    if not load:
        params = parseParameter(key, path_exp)

        if not os.path.exists(path_results + key):
            print('create result folder: ' + key)
            os.makedirs(path_results + key)

        if exp == 'Exp1':
            run_exp1(key,path_results,params)

        logging.debug('Simulations Done')

    else:
        # load results
        with open(path_results + load_key + os.sep + 'Results', 'rb') as f:
            results = pickle.load(f)
            logging.debug('Results loaded')
            f.close()

        parameters = {}
        with open(path_results + load_key + os.sep + 'Parameters', 'rb') as f:
            parameters = pickle.load(f)
            locals().update(parameters)
            logging.debug('Parameters loaded')
            f.close()

        logging.debug('Load Results Done')

#-------------------------------------------------------------------------------------------------------------------
#---- different experiments -------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------



def run_exp1(key, path_results,params):
    logging.debug('run experiment 1')

    N_sensors = params.get('N_sensors')
    min_beta = params.get('min_beta')
    max_beta = params.get('max_beta')
    SNR_db = params.get('SNR_db')
    init_type = params.get('init_type')
    Ny = params.get('Ny')
    Nz = params.get('Nz')
    target_rates = params.get('target_rates')

    if not result_check(path_results, key, 'Results'):

        beta_range = {'min': min_beta, 'max': max_beta}

        Nx = params.get('Nx')
        if Nx == 4:
            alphabet, p_x = create_alphabet('4-ASK')
        else:
            print('not implemented modulation scheme')

        p_xyi = create_channel_distribution(alphabet, p_x, SNR_db, N_sensors, Nx, Ny)

        # initial distributions for sensors
        if not 'init_type' in params:
            init_type = 'uniform'
        else:
            init_type = params.get('init_type')

        p_init, used_beta = create_init_quantizer(init_type, Ny, Nz, Nx, N_sensors, p_xyi)

        if used_beta == None:
            used_beta = [np.mean([min_beta, max_beta]) for _ in range(N_sensors)]
        params['initialization_beta'] = used_beta

        sib = SequentialDistributedIB(N_sensors, p_xyi, p_init, Nx, Ny, Nz, target_rates, beta_range, params)
        sib.run()
        results = sib.get_results()

        parameters = {'N_sensors': N_sensors, 'p_xyi': p_xyi, 'p_init': p_init, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'target_rates': target_rates, 'beta_range': beta_range, 'params': params}
        tmp = {'results': results, 'params': parameters}


        with open(path_results + key + os.sep + 'Results', 'wb') as f:
            pickle.dump(tmp, f)
            logging.debug('Results stored')
            f.close()
    else:
        print("Experiment already done!")

#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()