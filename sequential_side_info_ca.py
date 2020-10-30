import sys
import os
import numpy as np
import numpy.matlib


class SequentialDistributedIterativeCASideinformation():

    def __init__(self, N_sensors, optimization_order, p_x_yi, KL_init, p_init, Nx, Ny, Nz, pzbari_zi, accuracy, max_iter):
        self.opt_order = optimization_order
        self.N_sensors = N_sensors
        self.p_x_yi = p_x_yi
        self.p_x = np.sum(p_x_yi[0], 0)
        self.p_yi = [np.sum(x, 1) for x in self.p_x_yi]
        self.KL_init = KL_init
        self.p_init = p_init
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.pzbari_zi = pzbari_zi
        self.accuracy = accuracy
        self.max_iter = max_iter

    def run_one_iteration(self, sensor_idx, pzbari_yi, pzbari_x, iteration):

        # get j-th elements -> all other sensors than the i-th sensor
        j_elements = np.arange(0, self.N_sensors, dtype=int)
        j_elements = np.delete(j_elements, sensor_idx)

        # remove unused clusters
        Nz_storage = np.copy(self.Nz)
        z_used = [[] for _ in range(0, self.N_sensors)]
        for s_idx in range(0, self.N_sensors):
            z_used_tmp = []
            for z_idx in range(0, self.Nz[s_idx]):
                z_used_tmp = z_used_tmp + [z_idx]
            z_used[s_idx] = z_used_tmp

            pzbari_yi[s_idx] = np.copy(pzbari_yi[s_idx][z_used_tmp, :])
            pzbari_x[s_idx] = np.copy(pzbari_x[s_idx][z_used_tmp, :])
            self.Nz[s_idx] = len(z_used_tmp)

        # calculate p(zbar,x)
        p_zbar_x = np.ones(np.append(self.Nz, self.Nx))
        p_x_expand = np.tile(self.increase_dims(self.p_x, np.arange(0, self.N_sensors, dtype=int)),np.append(self.Nz, 1))
        for s_idx in range(0, self.N_sensors):
            # get j-th elements -> all other sensors than the i-th sensor
            j_ele = np.arange(0, self.N_sensors, dtype=int)
            j_ele = np.delete(j_ele, s_idx)

            pzbari_x_expand = np.tile(self.increase_dims(pzbari_x[s_idx] , j_ele), self.put_in_position(np.ones(self.N_sensors + 1, dtype=int), j_ele,np.array(self.Nz)[j_ele]))
            p_zbar_x = p_zbar_x * pzbari_x_expand
            pzbari_x_expand = None
        p_zbar_x = p_zbar_x * p_x_expand
        p_x_expand = None
        p_zbar = np.sum(p_zbar_x, -1)
        px_zbar = p_zbar_x / np.tile(np.expand_dims(p_zbar, -1), np.append(np.ones(self.N_sensors, dtype=int), self.Nx))

        # determine p(zbarjPartial|x)
        sensor_list_pos = np.where(np.array(self.opt_order, dtype=int) == sensor_idx)[0][0]
        partial_elements = self.opt_order[:sensor_list_pos + 1]
        partial_j_elements = self.opt_order[:sensor_list_pos]
        diff_node = np.setdiff1d(partial_elements, partial_j_elements)[0]
        diff_pos = self.dimension_difference(np.sort(partial_elements), np.sort(partial_j_elements))[0][0]
        remove_elements = self.opt_order[sensor_list_pos::]
        p_zbarjPartial_x = np.sum(p_zbar_x, tuple(remove_elements))
        p_zbarPartial_x = np.sum(p_zbar_x, tuple(remove_elements[1::]))
        pzbarjPartial_x = p_zbarjPartial_x / np.tile(self.increase_dims(self.p_x, np.arange(0, len(partial_j_elements), dtype=int)),np.append(np.array(self.Nz)[list(np.sort(np.array(partial_j_elements)))], 1))

        p_zbarjPartial_x = None
        p_zbar_x = None

        # determine p(zbarjPartial,yi,x)
        p_zbarPartial_yi_x = np.ones((list(np.array(self.Nz)[np.sort(partial_elements)]) + [self.Ny[sensor_idx]] + [self.Nx]))
        p_zbarPartial_yi_x = p_zbarPartial_yi_x * np.tile(self.increase_dims(pzbarjPartial_x, (diff_pos, len(partial_elements))),self.put_in_position(np.ones(len(partial_elements) + 2, dtype=int),(diff_pos, len(partial_elements)),(self.Nz[sensor_idx], self.Ny[sensor_idx])))
        tmp = [list(np.sort(partial_elements)).index(idx) for idx in partial_j_elements] + [-1]
        p_zbarPartial_yi_x = p_zbarPartial_yi_x * np.tile(self.increase_dims(pzbari_yi[sensor_idx], tmp),self.put_in_position(np.ones(len(partial_elements) + 2, dtype=int), tmp,np.append(np.array(self.Nz)[partial_j_elements],
                                                                               self.Nx)))
        p_zbarPartial_yi_x = p_zbarPartial_yi_x * np.tile( self.increase_dims(self.p_x_yi[sensor_idx], np.arange(0, len(partial_elements), dtype=int)),np.append(np.array(self.Nz)[np.sort(partial_elements)], np.ones(2, dtype=int)))
        p_zbarjPartial_yi_x = np.sum(p_zbarPartial_yi_x, diff_pos)

        # p(x|zbarjPartial,yi)
        px_zbarjPartialyi = p_zbarjPartial_yi_x / np.tile(np.expand_dims(np.sum(p_zbarjPartial_yi_x + 1e-31, -1), -1), np.append(np.ones(len(partial_j_elements) + 1, dtype=int), self.Nx))

        # p(x|zbarjPartial)
        px_zbarPartial = p_zbarPartial_x / np.tile(np.expand_dims(np.sum(p_zbarPartial_x + 1e-31, -1), -1), np.append(1, self.Nx))

        # determine extended Kullback-Leibler divergence
        KL = np.zeros(list(np.array(self.Nz)[np.sort(partial_elements)]) + [self.Ny[sensor_idx]])
        ptr1 = np.ones(len(partial_elements), dtype=int)
        ptr1[diff_pos] = self.Nz[sensor_idx]
        ptr2 = np.copy(np.array(self.Nz)[np.sort(partial_elements)])
        ptr2[diff_pos] = 1
        px_zbarjPartialyi_reshape = np.reshape(px_zbarjPartialyi, (-1, self.Nx))
        px_zbarPartial_reshape = np.reshape(px_zbarPartial, (-1, self.Nx))
        for runx in range(0, self.Nx):
            tmp1 = np.tile(np.reshape(px_zbarjPartialyi_reshape[:, runx], np.append(ptr2, self.Ny[sensor_idx])),np.append(ptr1, 1))
            tmp2 = np.tile(np.reshape(px_zbarPartial_reshape[:, runx], np.append(np.array(self.Nz)[np.sort(partial_elements)], 1)),np.append(np.ones(len(partial_elements), dtype=int), self.Ny[sensor_idx]))
            with np.errstate(all='ignore'):
                KL = KL + tmp1 * (np.log(tmp1) - np.log(tmp2))
            KL[np.isnan(KL)] = 0  # remove NaN
        tmp1 = None
        tmp2 = None
        px_zbarjPartialyi_reshape = None
        px_zbarPartial_reshape = None

        px_zbarjPartialyi = None
        px_zbarPartial = None

        pyi_x = [[] for _ in range(0, self.N_sensors)]
        for s_idx in range(0, self.N_sensors):
            pyi_x[s_idx] = self.p_x_yi[s_idx] / self.p_x

        p_x_y = np.ones(np.append(self.Ny, self.Nx))
        p_x_expand = np.tile(self.increase_dims(self.p_x, np.arange(0, self.N_sensors, dtype=int)),np.append(self.Ny, 1))
        for s_idx in range(0, self.N_sensors):
            # get j-th elements -> all other sensors than the i-th sensor
            j_ele = np.arange(0, self.N_sensors, dtype=int)
            j_ele = np.delete(j_ele, s_idx)
            pyi_x_expand = np.tile(self.increase_dims(pyi_x[s_idx], j_ele),self.put_in_position(np.ones(self.N_sensors + 1, dtype=int), j_ele,
                                                         np.array(self.Ny)[j_ele]))
            p_x_y = p_x_y * pyi_x_expand
            pyi_x_expand = None
        p_x_y = p_x_y * p_x_expand
        p_x_expand = None
        p_y = np.sum(p_x_y, -1)


        # determine p(yj|yi)
        p_y_partial = np.sum(p_y, tuple(remove_elements[1::]))
        p_yj_partial = np.sum(p_y, tuple(remove_elements))
        pyjpartial_yi = p_y_partial / np.tile(self.increase_dims(p_yj_partial, [diff_pos]),
                                                 self.put_in_position(np.ones(len(partial_elements), dtype=int),diff_pos, self.Ny[diff_node]))

        # determine p(zbarjPartial|yi)
        pzbarjPartial_yi = np.sum(p_zbarjPartial_yi_x, -1) / np.tile(self.increase_dims(self.p_yi[sensor_idx], np.arange(0, len(partial_j_elements), dtype=int)),np.append(np.array(self.Nz)[list(np.sort(np.array(partial_j_elements)))], 1))



        temporary = np.tile(np.expand_dims(pzbarjPartial_yi, diff_pos), self.put_in_position(np.ones(len(partial_j_elements) + 2, dtype=int), diff_pos, self.Nz[sensor_idx]))

        c_temp =   temporary * KL

        tmp_sum = [list(np.sort(partial_elements)).index(idx) for idx in partial_j_elements]

        c_temp = np.sum(c_temp, tuple(tmp_sum))

        C = np.tile(np.expand_dims(np.sum(self.pzbari_zi[sensor_idx], 0), -1), (1, self.Ny[sensor_idx])) * c_temp

        ptr = np.argmin(C, 0)
        pyjpartial_yi = None

        u = [[] for _ in range(0, self.N_sensors)]
        v = [[] for _ in range(0, self.N_sensors)]
        pzi_yi_tmp = np.zeros([self.Nz[sensor_idx], self.Ny[sensor_idx]])
        pzi_yi = [[] for _ in range(0, self.N_sensors)]
        for run_idx in range(0, self.Ny[sensor_idx]):
            # subindexes for Ny dimensions
            u = np.unravel_index(run_idx, self.Ny[sensor_idx])
            # subindexes for Nz dimensions
            v = np.unravel_index(ptr.flatten()[run_idx], self.Nz[sensor_idx])
            pzi_yi_tmp[v + u] = 1
        pzi_yi[sensor_idx] = pzi_yi_tmp
        pzbari_yi[sensor_idx] = np.sum( np.tile(np.expand_dims(self.pzbari_zi[sensor_idx], axis=2), (1, 1, self.Ny[sensor_idx])) *
                                        np.tile(np.expand_dims(pzi_yi[sensor_idx], axis=0),(self.Nz[sensor_idx], 1, 1)), axis=1)

        #update pmf

        p_zbari_yi_x = np.tile(np.expand_dims(self.p_x_yi[sensor_idx], 0), (self.Nz[sensor_idx], 1, 1)) * np.tile(np.expand_dims(pzbari_yi[sensor_idx], -1), self.Nx)
        pzbari_x[sensor_idx] = (1 / np.tile(np.expand_dims(self.p_x, 0), np.append(self.Nz[sensor_idx], 1))) * np.sum(p_zbari_yi_x, 1)
        p_zbari_yi_x = None

        p_zbar_x = np.ones(np.append(self.Nz, self.Nx))
        p_x_expand = np.tile(self.increase_dims(self.p_x, np.arange(0, self.N_sensors, dtype=int)),np.append(self.Nz, 1))
        for s_idx in range(0, self.N_sensors):
        #     # get j-th elements -> all other sensors than the i-th sensor
            j_ele = np.arange(0, self.N_sensors, dtype=int)
            j_ele = np.delete(j_ele, s_idx)

            pzbari_x_expand = np.tile(self.increase_dims(pzbari_x[s_idx] , j_ele), self.put_in_position(np.ones(self.N_sensors + 1, dtype=int), j_ele,np.array(self.Nz)[j_ele]))
            p_zbar_x = p_zbar_x * pzbari_x_expand
            pzbari_x_expand = None
        p_zbar_x = p_zbar_x * p_x_expand
        p_x_expand = None

        p_zbar = np.sum(p_zbar_x, -1)
        px_zbar = p_zbar_x / np.tile(np.expand_dims(p_zbar + 1e-31, -1), np.append(np.ones(self.N_sensors, dtype=int), self.Nx))
        #
        return pzbari_yi[sensor_idx], pzbari_x[sensor_idx], p_zbar_x, px_zbar, p_zbar


    # --------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------- helper functions ------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------


    def dimension_difference(self,shape1,shape2):
        if len(shape1) > len(shape2):
            diff_elements = np.setdiff1d(shape1,shape2)
            diff_pos = np.where(shape1 == diff_elements)
        elif len(shape1) < len(shape2):
            diff_elements = np.setdiff1d(shape2,shape1)
            diff_pos = np.where(shape2 == diff_elements)
        else:
            difference = []
            for idx in range(0,len(shape1)):
                    if shape1[idx] != shape2[idx]:
                        difference = difference+[idx]
            diff_pos = difference
        return diff_pos

    def put_in_position(self, x_, pos_, val_):
        newx = np.copy(x_)
        np.put(newx, pos_, val_)
        return newx

    def increase_dims(self, x_, dim_tuple_):
        # old_dimensions = x_.shape
        newx = np.copy(x_)
        dim_tuple_ = np.sort(dim_tuple_)
        for i in range(0, len(dim_tuple_)):
            newx = np.expand_dims(newx, dim_tuple_[i])
        return newx