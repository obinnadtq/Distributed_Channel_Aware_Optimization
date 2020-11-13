import sys
import os
import itertools
import numpy as np

path = os.getcwd()

#-----------------------------helpful functions---------------------------------
def replace_variational_parameter(dictionary,variational_parameter,var_value):
    # replace the variational value
    if dictionary.get(variational_parameter) != None:
        dictionary[variational_parameter] = var_value
    return dictionary

def store_params(exp_path, exp_file, dictionary):

    param_string_list = []
    for key in dictionary.keys():
        param_string_list.append(key + " = " +str(dictionary.get(key)))

    # create exp file
    with open(exp_path + os.sep + exp_file+'.exp', "a") as f:
        for line in param_string_list:
            f.write(line + '\n')
        f.write('\n')
        f.close()


    # remove previously defined variables
    dictionary = {}
    return dictionary

#----------------------------Path and name--------------------------------------
name = 'symmetric_3_nodes_scalar'

configuration = 'Configuration1'

exp = 'Exp1'

#-------------------------------------------------------------------------------
exp_path = path + os.sep + configuration + os.sep + exp + os.sep + 'Exp_files'

if not os.path.exists(exp_path):
    print('Exp file directory does not exist. Please create it before using this script')
    sys.exit()


#----------------------------Parameter------------------------------------------
# all parameter variables have to start with 'par_'

#define parameter for variation
# variational_parameter = 'sensor_opt_order'
# variational_parameter_values = [list(x) for x in list(itertools.permutations([0,1,2,3]))]

variational_parameter = 'target_rates'
variational_parameter_values = [[float("{:.2f}".format(x)),float("{:.2f}".format(x)),float("{:.2f}".format(x)),float("{:.2f}".format(x)),float("{:.2f}".format(x)),float("{:.2f}".format(x))] for x in np.arange(0.1,3,1/48)]

# variational_parameter = 'par_target_rates'
# variational_parameter_values = [[float("{:.2f}".format(x)),float("{:.2f}".format(x)),float("{:.2f}".format(x)),float("{:.2f}".format(x)),float("{:.2f}".format(x))] for x in np.arange(0.1,3,0.02)]

# variational_parameter = 'par_target_rates'
# variational_parameter_values = [[float("{:.5f}".format(x)),float("{:.5f}".format(x)),float("{:.5f}".format(x)),float("{:.5f}".format(x))] for x in np.arange(0.1,5,0.025)]

# variational_parameter = 'par_target_rates'
# variational_parameter_values = [[float("{:.5f}".format(x)),float("{:.5f}".format(x)),float("{:.5f}".format(x))] for x in np.arange(0.1,5,1/30)]

# variational_parameter = 'par_target_rates'
# variational_parameter_values = [[float("{:.5f}".format(x)),float("{:.5f}".format(x))] for x in np.arange(0.1,5,0.025)]

for var_idx in range(0,len(variational_parameter_values)):

    # filename
    exp_file = name + '_' + str(variational_parameter_values[var_idx])
    exp_file = "".join(x for x in exp_file if (x.isalnum() or x in "_-"))
    if not os.path.exists(exp_path + os.sep + exp_file + '.exp'):
        par = {}

        N_sensors = 6
        par['N_sensors'] = N_sensors
        par['Nx'] = 4
        par['Ny'] = [64, 64, 64, 64, 64, 64]
        par['Nz'] = [4, 4, 4, 4, 4, 4]
        par['Nstar'] = 12
        par['SNR_db'] = [8, 8, 8, 8, 8, 8]
        # par['SNR_db'] = [5, 5, 5, 5, 5, 5]
        # par['SNR_db'] = [3, 3, 3, 3, 3, 3]


        # N_sensors = 5
        # par['N_sensors'] = N_sensors
        # par['Nx'] = 4
        # par['Ny'] = [64, 64, 64, 64, 64]
        # par['Nz'] = [4, 4, 4, 4, 4]
        # par['Nstar'] = 12
        # par['SNR_db'] = [8, 8, 8, 8, 8]
        # # par['SNR_db'] = [5, 5, 5, 5, 5]
        # # par['SNR_db'] = [3, 3, 3, 3, 3]

        # N_sensors = 4
        # par['N_sensors'] = N_sensors
        # par['Nx'] = 4
        # par['Ny'] = [64, 64, 64, 64]
        # # par['Nz'] = [4, 4, 4, 4]
        # # par['Nz'] = [4, 4, 8, 8]
        # par['Nz'] = [8, 8, 4, 4]
        # par['Nstar'] = 12
        # # par['SNR_db'] = [8, 8, 8, 8]
        # # par['SNR_db'] = [5, 5, 5, 5]
        # # par['SNR_db'] = [3, 3, 3, 3]
        # par['SNR_db'] = [2, 4, 6, 8]

        # N_sensors = 3
        # par['N_sensors'] = N_sensors
        # par['Nx'] = 4
        # par['Ny'] = [64, 64, 64]
        # par['Nz'] = [4, 4, 4]
        # par['Nstar'] = 12
        # par['SNR_db'] = [8, 8, 8]
        # par['SNR_db'] = [5, 5, 5]
        # par['SNR_db'] = [3, 3, 3]

        # N_sensors = 2
        # par['N_sensors'] = N_sensors
        # par['Nx'] = 4
        # par['Ny'] = [64, 64]
        # par['Nz'] = [4, 4]
        # par['Nstar'] = 12
        # par['SNR_db'] = [8, 8]
        # par['SNR_db'] = [5, 5]
        # par['SNR_db'] = [3, 3]

        par = replace_variational_parameter(par, variational_parameter, variational_parameter_values[var_idx])
        par = store_params(exp_path, exp_file, par)   # to ensure the block structure

        par['sensor_opt_order'] = [x for x in range(0,N_sensors)]
        par['sensor_opt_with_side'] = [0 for x in range(0,N_sensors)]
        par['target_rates'] = [1.0 for x in range(0,N_sensors)]
        # par['target_rates'] = [1,1.5,2,2.5]
        # par['target_rates'] = [2.5, 2, 1.5, 1]
        par['min_beta'] = [0.01 for x in range(0,N_sensors)]
        par['max_beta'] = [100 for x in range(0,N_sensors)]
        par['accuracy_aib'] = 1e-3
        par['max_iter_aib'] = 1
        par['accuracy_bs'] = 1e-3
        par['max_iter_bs'] = 50
        par['accuracy_ib'] = 1e-10
        par['max_iter_ib'] = 100

        par = replace_variational_parameter(par, variational_parameter, variational_parameter_values[var_idx])
        par = store_params(exp_path, exp_file, par)  # to ensure the block structure

        par['init_type'] = "'uniform'"
        par['uniform_ib_sensor_init'] = True
        par['ib_solver'] = "'iterative'"
        par['beta_evolution'] = "'bisection'"
        par['ib_conv_crit'] = "'JS'"
        par['aib_conv_crit'] = "'JS'"

        par = replace_variational_parameter(par, variational_parameter, variational_parameter_values[var_idx])
        par = store_params(exp_path, exp_file, par)  # to ensure the block structure

        par['side_information'] = "'partial'"
        par['algorithm'] = "'aib'"
        par['compressed_side_info'] = False
        par['compression'] = "'sequential'"

        par = replace_variational_parameter(par, variational_parameter, variational_parameter_values[var_idx])
        par = store_params(exp_path, exp_file, par)  # to ensure the block structure

    else:
        print(exp_path + os.sep + exp_file+'.exp already exists!')

