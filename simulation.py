import numpy as np
import math
import matplotlib.pyplot as plt
from sequential_distributed_solution import SequentialDistributedCA

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

def create_forward_channel_distribution(N_sensors, Nz, SNR):
    pzbari_zi = [[] for _ in range(0, N_sensors)]
    SNR_lin = 10 ** (SNR / 10)
    for s_idx in range(0, N_sensors):
        temp = np.ones((Nz[s_idx], Nz[s_idx]))
        alphabet = np.arange(-(Nz[s_idx] - 1), Nz[s_idx] + 1, 2)
        p_z = 1 / Nz[s_idx] * np.ones(Nz[s_idx])
        SNR_norm = SNR_lin / (np.dot(alphabet ** 2, p_z))
        for idx1 in range(len(temp)):
            for idx2 in range(len(temp[s_idx])):
                if idx1 != idx2:
                    temp[idx1, idx2] = 0.5 * math.erfc(
                    np.sqrt((((alphabet[idx1] - alphabet[idx2]) / 2) ** 2) * SNR_norm))
        for j in range(len(temp)):
            for i in range(len(temp[j])):
                if i == j:
                    allElementsInRow = temp[:, j]
                    allElementsWithoutDiagonal = np.delete(allElementsInRow, [j])
                    temp[i, j] = temp[i, j] - np.sum(allElementsWithoutDiagonal)
        pzbari_zi[s_idx] = temp
    return pzbari_zi

def create_init_quantizer(KL_init, pzbari_zi, N_sensors, Nz, Ny):
    p_init = [[] for _ in range(0, N_sensors)]
    pbar_init = [[] for _ in range(0, N_sensors)]
    for i in range(0, N_sensors):
        p_tmp = np.zeros([Nz[i], Ny[i]])
        for run in range(0, Nz[i]):
            ptr = np.argmin(np.sum(np.tile(np.expand_dims(KL_init[i], axis=1), (1, Nz[i], 1))
                                   * np.tile(np.expand_dims(pzbari_zi[i], axis=2), (1, 1, Ny[i])), 0),axis=0)
            p_tmp[ptr, np.arange(ptr.size, dtype=int)] = 1
        p_init[i] = p_tmp
        pbar_init[i] = np.sum( np.tile(np.expand_dims(pzbari_zi[i], axis=2), (1, 1, Ny[i])) *
                                   np.tile(np.expand_dims(p_init[i], axis=0),(Nz[i], 1, 1)), axis=1)

    return pbar_init, p_init

def bestKL(Nz, Ny, Nx, pzbar_z, p_y, px_y, p_x_y, p_x):
    C_s = []
    best=[]
    for index in range(0, 200):
        KL_init = np.random.random((Nz, Ny))
        C_s.append(KL_init)
        Co = 1000
        convergence_params = 10 ** -4
        count = 0
        while True:
            ptr = np.argmin(np.sum(
                np.tile(np.expand_dims(KL_init, axis=1), (1, Nz, 1)) * np.tile(np.expand_dims(pzbar_z, axis=2),
                                                                                    (1, 1, Ny)),
                0), axis=0)
            pz_y = np.zeros((Nz, Ny))
            pz_y[ptr, np.arange(ptr.size, dtype=int)] = 1
            pzbar_y = np.sum(
                np.tile(np.expand_dims(pzbar_z, axis=2), (1, 1, Ny)) * np.tile(np.expand_dims(pz_y, axis=0),
                                                                               (Nz, 1, 1)), axis=1)
            p_zbar = np.sum(pzbar_y * p_y, axis=1)

            px_zbar = (1 / np.tile(np.expand_dims(p_zbar + 1e-31, axis=1), (1, Nx))) * np.sum(
                np.tile(np.expand_dims(p_x_y, axis=0),
                        (Nz, 1, 1)) * np.tile(
                    np.expand_dims(pzbar_y, axis=2), (1, 1, Nx)), axis=1)
            px_zbar_expanded = np.tile(np.expand_dims(px_zbar, axis=1), (1, Ny, 1))
            px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz, 1, 1))
            C_updated = np.sum((np.log2(px_y_expanded + 1e-31) - np.log2(px_zbar_expanded + 1e-31)) * px_y_expanded, 2)
            Cm = np.sum(p_y) * np.sum(pzbar_y * C_updated, 0)
            eff = (Co - (Cm + 1e-31)) / (Cm + 1e-31)
            if np.all(eff) <= convergence_params and count == 20:
                break
            else:
                Co = Cm
                C_init = C_updated
                count = count + 1
        p_x_zbar = px_zbar * np.tile(np.expand_dims(p_zbar, axis=1), (1, Nx))
        w = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_zbar, axis=1),
                                                                          (1, Nx))
        w1 = np.log2(p_x_zbar + 1e-31) - np.log2(w + 1e-31)
        I_x_zbar = np.sum(p_x_zbar * w1)
        best.append(I_x_zbar)
    Ixzbar = np.max(best)
    return C_s[best.index(Ixzbar)]

#------------------------------Run experiments-----------------------------------------------#

def run_exp2():
    N_sensors = 2
    SNR_db_sensors = [8, 8]
    Nx = 4
    Ny = [64, 64]
    Nz = [4, 4]
    alphabet = np.arange(-3, 5, 2)
    p_x = 1 / Nx * np.ones(Nx)
    p_x_yi = create_channel_distribution(alphabet, p_x, SNR_db_sensors, N_sensors, Nx, Ny)

    #create initial KL
    # KL_init = create_init_KL_divergence(N_sensors, Ny, Nz)
    pzbari_zi = create_forward_channel_distribution(N_sensors, Nz, 10)
    obi = bestKL(Nz[0], Ny[0], Nx, pzbari_zi[0], np.sum(p_x_yi[0], 1),p_x_yi[0] / np.tile(np.expand_dims(np.sum(p_x_yi[0], 1), -1), (1, Nx)), p_x_yi[0], p_x)
    KL_init = [obi, obi, obi]
    pbar_init, p_init = create_init_quantizer(KL_init, pzbari_zi, N_sensors, Nz, Ny)
    sib = SequentialDistributedCA(N_sensors, p_x_yi, KL_init, pbar_init, p_init, Nx, Ny, Nz, pzbari_zi)
    sib.run()
    results = sib.get_results()
    x_values = range(0, 2)
    y_values = np.zeros(2)
    y_values[0] = results.get('Init_sum_rate')
    last_node = [0, 1][-1]
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












