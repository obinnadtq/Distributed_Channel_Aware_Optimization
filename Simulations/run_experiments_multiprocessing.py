#!/usr/bin/env python

#python script to run all experiments of a specific experiment folder
#usage: run_experiment.py [configuration] [experiment] [number of parallel processes]

import sys
import os
import logging
from multiprocessing import Pool

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.abspath(''),'Configuration1'))

    sys.path.append(os.path.join(os.path.abspath('..'),'information_bottleneck','alternating_information_bottleneck_algorithms'))
    sys.path.append(os.path.join(os.path.abspath('..'),'information_bottleneck','alternating_information_bottleneck_algorithms','generic_classes'))
    sys.path.append(os.path.join(os.path.abspath('..'),'information_bottleneck','alternating_information_bottleneck_algorithms','scalar_IB'))
    sys.path.append(os.path.join(os.path.abspath('..'),'information_bottleneck','alternating_information_bottleneck_algorithms','sIB'))
    sys.path.append(os.path.join(os.path.abspath('..'),'information_bottleneck','ib_base','information_bottleneck_algorithms'))
    sys.path.append(os.path.join(os.path.abspath('..'),'information_bottleneck','ib_base','tools'))

    import simulation_c1


    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) == 1:
        logging.error('No arguments given! -> usage: run_experiment.py [configuration] [experiment] [number of parallel processes][process_command(e.g. python3)]')
        sys.exit()
    conf = sys.argv[1]
    if conf == None:
        logging.error('No configuration given! -> usage: run_experiment.py [configuration] [experiment] [number of parallel processes][process_command(e.g. python3)]')
        sys.exit()
    exp = sys.argv[2]
    if exp == None:
        logging.error('No experiment given! -> usage: run_experiment.py [configuration] [experiment] [number of parallel processes][process_command(e.g. python3)]')
        sys.exit()
    run_in_parallel = int(sys.argv[3])
    if run_in_parallel == None:
        logging.error('How many processes in parallel? -> usage: run_experiment.py [configuration] [experiment] [number of parallel processes][process_command(e.g. python3)]')
        sys.exit()

    path_exp = os.getcwd()+os.sep+conf+os.sep+exp+os.sep+'Exp_files'+os.sep
    path_exec = os.getcwd()+os.sep+conf

    #get list of all files in directory
    exp_list = os.listdir(path_exp)
    experiment = [[] for _ in range(0,len(exp_list))]
    for exp_idx in range(0,len(exp_list)):
        experiment[exp_idx] = (conf+os.sep+exp,exp_list[exp_idx].split('.')[0])

    p = Pool(run_in_parallel)

    if conf == 'Configuration1':
        results = p.starmap(simulation_c1.main, experiment)

    logging.info('All experiments done ::: ')