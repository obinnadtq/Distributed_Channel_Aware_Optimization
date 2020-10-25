import numpy as np
import matplotlib.pyplot as plt
from sequential_distributed_solution import SequentialDistributedCA

#---------------------------------------helpful functions---------------------------------#
def create_channel_distribution(alphabet, p_x, SNR_db_sensors, N_sensors, Nx, Ny):
    # joint dist p_x_y
    # power of AWGN at sensor nodes
    SNR_lin = [np.power(10, (x / 10)) for x in SNR_db_sensors]
    sigma2_N = np.tile(np.dot(np.power(alphabet, 2), p_x), (1, N_sensors)) / SNR_lin

    # seperate p(x,yi) for every sensor
    p_x_yi = [[] for _ in range(0, N_sensors)]
    for s_idx in range(0,N_sensors):
        dy = 2 * (np.max(alphabet) + 5 * np.sqrt(sigma2_N[0, s_idx])) / Ny[s_idx]
        y = np.around(np.arange(-(np.max(alphabet) + 5 * np.sqrt(sigma2_N[0, s_idx])), np.max(alphabet) + 5 * np.sqrt(sigma2_N[0, s_idx]), dy), 4)
        p_x_yi[s_idx] = np.ones((Nx, Ny[s_idx]))
        for x_idx in range(0,Nx):
            p_x_yi[s_idx][x_idx,:] = np.exp(-np.power((y[np.arange(0, Ny[s_idx])] - alphabet[x_idx]), 2) / 2 / sigma2_N[0, s_idx]) * p_x[x_idx]
        p_x_yi[s_idx] = p_x_yi[s_idx] / np.sum(p_x_yi[s_idx])
        p_x_yi[s_idx] = np.transpose(p_x_yi[s_idx])
    return p_x_yi


def create_init_KL_divergence(N_sensors, Ny, Nz):
    KL_init = [[] for _ in range(0, N_sensors)]
    for i in range(0, N_sensors):
        KL_init[i] = np.random.random((Nz[i], Ny[i]))
    return KL_init

def create_forward_channel_distribution(N_sensors, Nz, Pe):
    pzbari_zi = [[] for _ in range(0, N_sensors)]
    for s_idx in range(0, N_sensors):
        pzbari_zi[s_idx] = np.ones((Nz[s_idx], Nz[s_idx]))
        for idx1 in range(len(pzbari_zi[s_idx])):
            for idx2 in range(len(pzbari_zi[s_idx])):
                if idx1 == idx2:
                    pzbari_zi[s_idx][idx1, idx2] = 1 - Pe
                else:
                    pzbari_zi[s_idx][idx1, idx2] = Pe / (Nz[s_idx] - 1)
    return pzbari_zi

def create_init_quantizer(KL_init, pzbari_zi, N_sensors, Nz, Ny):
    p_init = [[] for _ in range(0, N_sensors)]
    pbar_init = [[] for _ in range(0, N_sensors)]
    for i in range(0, N_sensors):
        N_ones = np.floor(Ny[i] / Nz[i])
        p_tmp = np.zeros([Nz[i], Ny[i]])
        for run in range(0, Nz[i]):
            # ptr = np.argmin(np.sum(np.tile(np.expand_dims(KL_init[i], axis=1), (1, Nz[i], 1))
            #                        * np.tile(np.expand_dims(pzbari_zi[i], axis=2), (1, 1, Ny[i])), 0),axis=0)
            # p_init[i][ptr, np.arange(ptr.size, dtype=int)] = 1
            ptr = range(run * int(N_ones), min(((run + 1) * int(N_ones)), Ny[i]))
            p_tmp[run, ptr] = 1
        if ptr[-1] < Ny[i]:
            p_tmp[run, range(ptr.stop, np.size(p_tmp, 1))] = 1
        p_init[i] = p_tmp
        pbar_init[i] = np.sum( np.tile(np.expand_dims(pzbari_zi[i], axis=2), (1, 1, Ny[i])) *
                                   np.tile(np.expand_dims(p_init[i], axis=0),(Nz[i], 1, 1)), axis=1)

    return pbar_init, p_init

#------------------------------Run experiments-----------------------------------------------#

def run_exp2():
    N_sensors = 3
    SNR_db_sensors = [8, 8, 8]
    Nx = 4
    Ny = [64, 64, 64]
    Nz = [4, 4, 4]
    alphabet = np.arange(-3, 5, 2)
    p_x = 1 / Nx * np.ones(Nx)
    p_x_yi = create_channel_distribution(alphabet, p_x, SNR_db_sensors, N_sensors, Nx, Ny)

    #create initial KL
    KL_init = create_init_KL_divergence(N_sensors, Ny, Nz)

    Pe = 0.4
    pzbari_zi = create_forward_channel_distribution(N_sensors, Nz, Pe)
    pbar_init, p_init = create_init_quantizer(KL_init, pzbari_zi, N_sensors, Nz, Ny)

    sib = SequentialDistributedCA(N_sensors, p_x_yi, KL_init, pbar_init, p_init, Nx, Ny, Nz, pzbari_zi)
    sib.run()
    results = sib.get_results()
    x_values = range(0, 2)
    y_values = np.zeros(2)
    y_values[0] = results.get('Init_sum_rate')
    last_node = [0, 1, 2][-1]
    for sca_idx in range(0, 1):
        y_values[sca_idx + 1] = results['iterations_sca'][sca_idx]['Node_results'][last_node]['Iterations_bs'][49]['I_x_z']
    plt.plot(x_values, y_values, '-+', label='sCA_optimized')
    plt.xticks(np.linspace(0, 1, 2, dtype=int))
    plt.title('Vector I(x;zbar) versus iterations of sCA after optimizing every node')
    plt.ylabel('I(x;zbar)')
    plt.xlabel('Iteration sCA')
    plt.legend()
    plt.grid()
    plt.show()







run_exp2()








