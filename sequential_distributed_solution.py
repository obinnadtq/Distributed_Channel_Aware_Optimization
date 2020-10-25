from sequential_side_info_ca import SequentialDistributedIterativeCASideinformation

import numpy as np
class SequentialDistributedCA():
    def __init__(self, N_sensors, p_x_yi, KL_init, pbar_init, p_init, Nx, Ny, Nz, pzbari_zi):
        self.N_sensors = N_sensors
        self.p_x_yi = p_x_yi
        self.p_x = np.sum(p_x_yi[0], 0)
        self.p_yi = [np.sum(x, 1) for x in self.p_x_yi]
        self.KL_init = KL_init
        self.pbar_init = pbar_init
        self.p_init = p_init
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.pzbari_zi = pzbari_zi
        self.name = 'SCA'
        self.sensor_opt_order = [0, 1, 2]
        self.accuracy_ca = 1e-10
        self.max_iter_ca = 100
        self.accuracy_sca = 0.001
        self.max_iter_sca = 1
        self.max_iterations_bs = 50

        for idx in range(0, self.N_sensors):
            self.solver_sideinfo = SequentialDistributedIterativeCASideinformation(self.N_sensors, self.sensor_opt_order,
                                                                               self.p_x_yi, self.KL_init, self.p_init, self.Nx, self.Ny,
                                                                               self.Nz, self.pzbari_zi, self.accuracy_ca, self.max_iter_ca)
        sca_result = {}
        sca_result['iterations_sca'] = [[] for _ in range(0,self.max_iter_sca)]

        for sca_idx in range(0, self.max_iter_sca):
            sca_result['iterations_sca'][sca_idx] = {}
            sca_result['iterations_sca'][sca_idx]['residual_sca'] = None

            node_results = [[] for _ in range(0, self.N_sensors)]
            for node_idx in range(0, self.N_sensors):
                node_results[node_idx] = {}

                node_bs_iteration_results = [[] for _ in range(0, self.max_iterations_bs)]
                for bs_idx in range(0, self.max_iterations_bs):
                    node_bs_iteration_results[bs_idx] = {}
                    node_bs_iteration_results[bs_idx]['residual_ca'] = [None for _ in range(0, self.max_iter_ca)]
                    node_bs_iteration_results[bs_idx]['used_iter_ca'] = None
                    node_bs_iteration_results[bs_idx]['I_x_z'] = None

                node_results[node_idx]['Iterations_bs'] = node_bs_iteration_results
            sca_result['iterations_sca'][sca_idx]['Node_results'] = node_results
        self.sca_result = sca_result


    def run(self):
        self.run_algorithm()

    def run_algorithm(self):
        self.determine_sum_rate(self.pbar_init, init=True)
        iter_sca = 0
        residual_sca = 1
        KLi = self.KL_init
        pzbari_yi = np.copy(self.pbar_init)

        while((residual_sca > self.accuracy_sca) and (iter_sca < self.max_iter_sca)):
            pzbari_yi_prev = np.copy(pzbari_yi)
            pzbari_x= self.determine_pzbari_x(pzbari_yi)

            for sensor_idx in self.sensor_opt_order:
            #-----------------channel aware algorithm--------------------------------------------#
                count = 0
                while(count < self.max_iterations_bs):
                    iter_ca = 0
                    residual_ca = 1
                    pzbari_yi[sensor_idx] = np.copy(self.pbar_init[sensor_idx])
                    pzbari_x[sensor_idx] = self.determine_pzbari_x(pzbari_yi, sensor_idx)
                    while(( residual_ca > self.accuracy_ca) and (iter_ca < self.max_iter_ca)):
                        pzbari_yi_prev_ca = np.copy(pzbari_yi[sensor_idx])

                        pzbari_yi[sensor_idx], pzbari_x[sensor_idx], p_zbar_x, px_zbar, p_zbar =  self.solver_sideinfo.run_one_iteration(sensor_idx, pzbari_yi, pzbari_x, iter_ca)

                        residual_ca = self.calculate_residual_CA(pzbari_yi[sensor_idx], pzbari_yi_prev_ca, sensor_idx)
                        iter_ca += 1
                    self.determine_mutual_info(pzbari_yi, pzbari_x, p_zbar_x, px_zbar, p_zbar, sensor_idx, iter_sca, count)
                    count += 1

            residual_sca = self.calculate_residual_sCA(pzbari_yi, pzbari_yi_prev)
            iter_sca += 1
    #-----------------------------helper functions------------------------------------------------------#
    def calculate_residual_CA(self, Q, Q_prev,sensor_idx):
        residual_ca = np.sum(self.res_JSD(Q, Q_prev, [0.5, 0.5]) * self.p_yi[sensor_idx])
        return residual_ca

    def put_in_position(self, x_, pos_, val_):
        newx = np.copy(x_)
        np.put(newx, pos_, val_)
        return newx

    def res_KLD(self, Q1, Q2):
        """ Return a vector for y of the KL Divergence for two input quantizers"""
        with np.errstate(all='ignore'):
            tmp = np.log2(Q1) - np.log2(Q2)
            tmp[np.isnan(tmp)] = 0
            tmp[np.isinf(tmp)] = 0
            KL = np.sum(Q1 * tmp, axis=0)
            KL[np.isnan(KL)] = 0  # remove NaN
        return KL

    def res_JSD(self, Q1, Q2, pi_weigthts):
        """ Return a vector for y of the JS Divergence for two input quantizers"""
        p_bar = pi_weigthts[0] * Q1 + pi_weigthts[1] * Q2
        KL1 = self.res_KLD(Q1, p_bar)
        KL2 = self.res_KLD(Q2, p_bar)
        return pi_weigthts[0] * KL1 + pi_weigthts[1] * KL2

    def get_results(self):
        return self.sca_result

    def calculate_residual_sCA(self, Q_all, Q_all_prev):
        residual_sca = 0
        for sensor_idx in range(0, self.N_sensors):
            tmp = self.res_JSD(Q_all[sensor_idx], Q_all_prev[sensor_idx], [0.5, 0.5]) * self.p_yi[sensor_idx]
            residual_sca = residual_sca + np.sum(tmp)
        residual_sca = residual_sca / self.N_sensors
        return residual_sca

    def determine_pzbari_x(self, pzbari_yi, sensor_idx=None):

        pzbari_x = [[] for _ in range(0, self.N_sensors)]
        for idx in range(0, self.N_sensors):
            p_ziyixi = np.tile(np.expand_dims(self.p_x_yi[idx], 0), np.append(self.Nz[idx], np.ones(2, dtype=int))) * \
                       np.tile(np.expand_dims(pzbari_yi[idx], -1), self.Nx)
            pzbari_x[idx] = (1 / np.tile(np.expand_dims(self.p_x, 0), np.append(self.Nz[idx], 1))) * np.sum(p_ziyixi, 1)
        if sensor_idx == None:
            return pzbari_x
        else:
            return pzbari_x[sensor_idx]

    def determine_sum_rate(self, pzbari_yi, init = False, pzbari_x = None, sensor_idx = None, iter_sca = None, p_zbar_x = None, count = None, px_zbar = None, iter_ca = None, scalar = False):
        if init:
            pzbari_x = self.determine_pzbari_x(pzbari_yi)
            for idx in range(0, self.N_sensors):
                p_zbari_yi_xi = np.tile(np.expand_dims(self.p_x_yi[idx], 0),np.append(self.Nz[idx], np.ones(2, dtype=int))) * np.tile(np.expand_dims(pzbari_yi[idx], -1), self.Nx)
                pzbari_x[idx] = (1 / np.tile(np.expand_dims(self.p_x, 0), np.append(self.Nz[idx], 1))) * np.sum(p_zbari_yi_xi, 1)

            p_zbar_x = np.ones(np.append(self.Nz, self.Nx))
            p_x_expand = np.tile(self.increase_dims(self.p_x, np.arange(0, self.N_sensors, dtype=int)),np.append(self.Nz, 1))
            for s_idx in range(0, self.N_sensors):
                # get j-th elements -> all other sensors than the i-th sensor
                j_ele = np.arange(0, self.N_sensors, dtype=int)
                j_ele = np.delete(j_ele, s_idx)
                pzbari_x_expand = np.tile(self.increase_dims(pzbari_x[s_idx], j_ele),self.put_in_position(np.ones(self.N_sensors + 1, dtype=int),
                                                                                                          j_ele,np.array(self.Nz)[j_ele]))
                p_zbar_x = p_zbar_x * pzbari_x_expand
                pzbari_x_expand = None
            p_zbar_x = p_zbar_x * p_x_expand
            p_x_expand = None
            p_zbar = np.sum(p_zbar_x, -1)
            px_zbar = p_zbar_x / np.tile(np.expand_dims(p_zbar, -1), np.append(np.ones(self.N_sensors, dtype=int), self.Nx))

        #determine I(X;Zbar)
        with np.errstate(all='ignore'):
            tmp1 = np.log2(self.p_x)
            tmp1[np.isnan(tmp1)] = 0  # remove NaN
            tmp2 = np.log2(px_zbar)
            tmp2[np.isnan(tmp2)] = 0  # remove NaN
            tmp = p_zbar_x * (tmp2 - np.tile(tmp1, np.append(self.Nz, 1)))
            tmp[np.isnan(tmp)] = 0  # remove NaN
        if init:
            self.sca_result['Init_sum_rate'] = np.sum(tmp)
        else:
            self.sca_result['iterations_sca'][iter_sca]['Node_results'][sensor_idx]['Iterations_bs'][count]['I_x_z'] = np.sum(tmp)
        tmp = None
        tmp1 = None
        tmp2 = None

    def increase_dims(self, x_, dim_tuple_):
        newx = np.copy(x_)
        dim_tuple_ = np.sort(dim_tuple_)
        for i in range(0, len(dim_tuple_)):
            newx = np.expand_dims(newx, dim_tuple_[i])
        return newx

    def determine_mutual_info(self, pzbari_yi, pzbari_x, p_zbar_x, px_zbar, p_zbar, sensor_idx, iter_sca, count, scalar= False, iter_ca = None):
        self.determine_sum_rate(pzbari_yi, init=False, pzbari_x = pzbari_x, sensor_idx = sensor_idx, iter_sca = iter_sca,p_zbar_x = p_zbar_x, px_zbar = px_zbar, iter_ca = iter_ca, count = count
                                )
