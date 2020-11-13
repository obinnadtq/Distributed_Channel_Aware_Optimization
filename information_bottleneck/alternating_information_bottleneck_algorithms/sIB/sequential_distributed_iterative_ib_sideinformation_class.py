import sys
import os
import numpy as np
import numpy.matlib
import math

from generic_IB_sideinformation_class import GenericIBSideinformation

__author__ = "Steffen Steiner"
__copyright__ = "12.11.2018, Institute of Communications, University of Rostock "
__credits__ = ["Steffen Steiner"]
__version__ = "1.0"
__email__ = "steffen.steiner@uni-rostock.de"
__status__ = "Release"
__doc__="""This module contains the iterative Information Bottleneck algorithm for multiple sensors"""


class SequentialDistributedIterativeIBSideinformation(GenericIBSideinformation):
    """This class can be used to perform the iterative Information Bottleneck algorithm for multiple sensors.
        Args:
    input parameter
        Nsensors                number of sensors
        p_xyi                    input joint pdf for x and yi -> first dimension x, seccond dimension y
        p_init                  initial pmf p_zi_yi for all sensors
        Nx                      cardinality of x
        Ny                      cardinality of y for all sensors
        Nz                      cardinality of z for all sensors
        beta                    beta the specific sensor
        accuracy                accuracy for all sensors
        max_iter                maximum number of iterations of ib algorithm
    output PDF_s
        p_zi_yi
        p_zi_x
        p_zx
        p_x_z
        p_z

    NOTE:   -> pmf matrices are always arranged such that first variable is z, followed by y and x
            -> yi is a scalar and means the i-th (index of sensor_idx) value of y
            -> yj is a vector with all other y values except yi

    """

    def __init__(self, Nsensors_, optimization_order_, p_xyi_, p_init_, Nx_, Ny_, Nz_, beta_ , accuracy_, max_iter_):
        GenericIBSideinformation.__init__(self,Nsensors_, None, p_xyi_, p_init_, Nx_, Ny_, Nz_, beta_, accuracy_, max_iter_)
        self.opt_order = optimization_order_
        self.name = 'Iterative IB for multiple sensors'

    def run(self):
        raise RuntimeError('This function is not implemented in this class')

    def run_one_iteration(self,sensor_idx, p_zi_yi, p_zi_x, beta,iteration, side_information):

        if not side_information == 'partial':
            print('just implemented for partial sideinformation')
            sys.exit()

        # get j-th elements -> all other sensors than the i-th sensor
        # j_elements = np.arange(0, self.Nsensors, dtype=int)
        # j_elements = np.delete(j_elements, sensor_idx)

        j_elements = self.get_j_elements(sensor_idx)

        # remove unused clusters
        Nz_storage = np.copy(self.Nz)
        z_used = [[] for _ in range(0, self.Nsensors)]
        p_zi_yi_tmp = [[] for _ in range(0,self.Nsensors)]
        p_zi_x_tmp = [[] for _ in range(0, self.Nsensors)]
        for s_idx in range(0, self.Nsensors):
            z_used_tmp = []
            for z_idx in range(0, self.Nz[s_idx]):
                if not all(p_zi_yi[s_idx][z_idx, :] == 0):
                    z_used_tmp = z_used_tmp + [z_idx]
            z_used[s_idx] = z_used_tmp

            p_zi_yi_tmp[s_idx] = np.copy(p_zi_yi[s_idx][z_used_tmp, :])
            p_zi_x_tmp[s_idx] = np.copy(p_zi_x[s_idx][z_used_tmp, :])
            pzbari_zi = self.create_forward_channel_distribution(self.Nsensors, self.Nz, 12)
            p_zi_x_tmp[s_idx] = np.sum(np.tile(np.expand_dims(p_zi_x_tmp[s_idx], 0), (8, 1, 1)) * np.tile(
                np.expand_dims(pzbari_zi[s_idx], -1), (1, 1, 4)), 1)
            self.Nz[s_idx] = len(z_used_tmp)
        p_zi_yi = np.copy(p_zi_yi_tmp)
        p_zi_x = np.copy(p_zi_x_tmp)

        # calculate p(z,x)
        p_zx = np.ones(np.append(self.Nz, self.Nx))
        p_x_expand = np.tile(self.increase_dims(self.p_x, np.arange(0, self.Nsensors, dtype=int)), np.append(self.Nz, 1))
        for s_idx in range(0, self.Nsensors):
            # get j-th elements -> all other sensors than the i-th sensor
            j_ele = np.arange(0, self.Nsensors, dtype=int)
            j_ele = np.delete(j_ele, s_idx)

            p_zi_x_expand = np.tile(self.increase_dims(p_zi_x[s_idx], j_ele), self.put_in_position(np.ones(self.Nsensors + 1, dtype=int), j_ele, np.array(self.Nz)[j_ele]))
            p_zx = p_zx * p_zi_x_expand
            p_zi_x_expand = None
        p_zx = p_zx * p_x_expand
        p_x_expand = None

        p_z = np.sum(p_zx, -1)
        p_x_z = p_zx / np.tile(np.expand_dims(p_z, -1), np.append(np.ones(self.Nsensors, dtype=int), self.Nx))

        # determine p(zjPartial|x)
        sensor_list_pos = np.where(np.array(self.opt_order, dtype=int) == sensor_idx)[0][0]
        partial_elements = self.opt_order[:sensor_list_pos + 1]
        partial_j_elements = self.opt_order[:sensor_list_pos]
        diff_node = np.setdiff1d(partial_elements, partial_j_elements)[0]
        diff_pos = self.dimension_difference(np.sort(partial_elements), np.sort(partial_j_elements))[0][0]
        remove_elements = self.opt_order[sensor_list_pos::]
        p_zjPartialx = np.sum(p_zx, tuple(remove_elements))
        p_zPartialx = np.sum(p_zx, tuple(remove_elements[1::]))
        p_zjPartial_x = p_zjPartialx / np.tile(self.increase_dims(self.p_x,np.arange(0,len(partial_j_elements),dtype=int)),np.append(np.array(self.Nz)[list(np.sort(np.array(partial_j_elements)))],1))

        p_zjPartialx = None
        p_zx = None

        # determine p(zjPartial,yi,x)
        p_zPartialyix = np.ones((list(np.array(self.Nz)[np.sort(partial_elements)]) + [self.Ny[sensor_idx]] + [self.Nx]))
        p_zPartialyix = p_zPartialyix * np.tile(self.increase_dims(p_zjPartial_x, (diff_pos, len(partial_elements))),
                                  self.put_in_position(np.ones(len(partial_elements) + 2, dtype=int), (diff_pos, len(partial_elements)), (self.Nz[sensor_idx], self.Ny[sensor_idx])))
        tmp = [list(np.sort(partial_elements)).index(idx) for idx in partial_j_elements] +[-1]
        p_zPartialyix = p_zPartialyix * np.tile(self.increase_dims(p_zi_yi[sensor_idx], tmp),
                                  self.put_in_position(np.ones(len(partial_elements) + 2, dtype=int), tmp, np.append(np.array(self.Nz)[partial_j_elements], self.Nx)))
        p_zPartialyix = p_zPartialyix * np.tile(self.increase_dims(self.p_yix[sensor_idx], np.arange(0, len(partial_elements), dtype=int)), np.append(np.array(self.Nz)[np.sort(partial_elements)], np.ones(2, dtype=int)))
        p_zjPartialyix = np.sum(p_zPartialyix, diff_pos)


        # p(x|zjPartial,yi)
        p_x_zjPartialyi = p_zjPartialyix / np.tile(np.expand_dims(np.sum(p_zjPartialyix, -1), -1), np.append(np.ones(len(partial_j_elements) + 1, dtype=int), self.Nx))

        # p(x|zPartial)
        p_x_zPartial = p_zPartialx / np.tile(np.expand_dims(np.sum(p_zPartialx,-1), -1), np.append(1,self.Nx))

        # determine extended Kullback-Leibler divergence
        KL = np.zeros(list(np.array(self.Nz)[np.sort(partial_elements)]) + [self.Ny[sensor_idx]])
        ptr1 = np.ones(len(partial_elements) , dtype=int)
        ptr1[diff_pos] = self.Nz[sensor_idx]
        ptr2 = np.copy(np.array(self.Nz)[np.sort(partial_elements)])
        ptr2[diff_pos] = 1
        p_x_zjPartialyi_reshape = np.reshape(p_x_zjPartialyi, (-1, self.Nx))
        p_x_zPartial_reshape = np.reshape(p_x_zPartial, (-1, self.Nx))
        for runx in range(0, self.Nx):
            tmp1 = np.tile(np.reshape(p_x_zjPartialyi_reshape[:, runx], np.append(ptr2, self.Ny[sensor_idx])), np.append(ptr1, 1))
            tmp2 = np.tile(np.reshape(p_x_zPartial_reshape[:, runx], np.append(np.array(self.Nz)[np.sort(partial_elements)], 1)), np.append(np.ones(len(partial_elements), dtype=int), self.Ny[sensor_idx]))
            with np.errstate(all='ignore'):
                KL = KL + tmp1 * (np.log(tmp1) - np.log(tmp2))
            KL[np.isnan(KL)] = 0  # remove NaN
        tmp1 = None
        tmp2 = None
        p_x_zjPartialyi_reshape = None
        p_x_zPartial_reshape = None

        p_x_zjPartialyi = None
        p_x_zPartial = None

        # determine p(zjPartial|yi)
        p_zjPartial_yi = np.sum(p_zjPartialyix, -1) / np.tile(self.increase_dims(self.p_yi[sensor_idx], np.arange(0, len(partial_j_elements), dtype=int)), np.append(np.array(self.Nz)[list(np.sort(np.array(partial_j_elements)))], 1))


        # determine p(zi|zj)
        p_z_partial = np.sum(p_z, tuple(remove_elements[1::]))
        p_zj_partial = np.sum(p_z, tuple(remove_elements))
        p_zi_zj_partial = p_z_partial / np.tile(self.increase_dims(p_zj_partial, [diff_pos]), self.put_in_position(np.ones(len(partial_elements), dtype=int), diff_pos, self.Nz[diff_node]))


        # determine d(zi,yi)
        if np.isinf(beta):
            d_ziyi = KL * np.tile(np.expand_dims(p_zjPartial_yi, diff_pos), self.put_in_position(np.ones(len(partial_j_elements) + 2, dtype=int), diff_pos, self.Nz[sensor_idx]))
        else:
            d_ziyi = ((1 / beta) * KL - np.log(np.tile(np.expand_dims(p_zi_zj_partial,-1),np.append(np.ones(len(partial_elements),dtype=int),self.Ny[sensor_idx])))) * \
                     np.tile(np.expand_dims(p_zjPartial_yi, diff_pos), self.put_in_position(np.ones(len(partial_j_elements) + 2, dtype=int), diff_pos, self.Nz[sensor_idx]))

        tmp = [list(np.sort(partial_elements)).index(idx) for idx in partial_j_elements]
        d_ziyi = np.sum(d_ziyi, tuple(tmp))  # sum over all Zj j~=run_sensors
        p_z_partial = None
        p_zi_zj_partial = None
        p_z = None
        p_zj_yi = None
        p_zyix = None

        # check if distance gets too large -> then use deterministic solution
        deterministic_solution = False
        if any(np.min(d_ziyi, 0) > 500):
            deterministic_solution = True

        # update conditional pmfs of quantizers
        if np.isinf(beta) or deterministic_solution:
            # deterministic quantizer
            ptr = np.argmin(d_ziyi, axis=0)
            # u = [[] for _ in range(0, self.Nsensors)]
            # v = [[] for _ in range(0, self.Nsensors)]
            p_zi_yi_tmp = np.zeros([self.Nz[sensor_idx], self.Ny[sensor_idx]])
            for run_idx in range(0, self.Ny[sensor_idx]):
                # subindexes for Ny dimensions
                u = np.unravel_index(run_idx, self.Ny[sensor_idx])
                # subindexes for Nz dimensions
                v = np.unravel_index(ptr.flatten()[run_idx], self.Nz[sensor_idx])
                p_zi_yi_tmp[v + u] = 1
            p_zi_yi[sensor_idx] = p_zi_yi_tmp
        else:
            # update non-deterministic mapping
            p_zi_yi[sensor_idx] = np.exp(-d_ziyi)
            p_zi_yi[sensor_idx] = p_zi_yi[sensor_idx] / np.tile(np.sum(p_zi_yi[sensor_idx], 0), np.append(self.Nz[sensor_idx], 1))  # normalize pmf
            p_zi_yi[sensor_idx][np.isnan(p_zi_yi[sensor_idx])] = 0  # remove NaN

        # recover original dimensionality
        self.Nz = list(np.copy(Nz_storage))
        p_zi_yi_ext = [[] for _ in range(0,self.Nsensors)]
        p_zi_x_ext = [[] for _ in range(0,self.Nsensors)]
        for s_idx in range(0, self.Nsensors):
            p_zi_yi_tmp = np.zeros([self.Nz[s_idx],self.Ny[s_idx]])
            p_zi_yi_tmp[z_used[s_idx], :] = np.copy(p_zi_yi[s_idx])
            p_zi_yi_ext[s_idx] = np.copy(p_zi_yi_tmp)

            p_zi_x_tmp = np.zeros([self.Nz[s_idx], self.Nx])
            p_zi_x_tmp[z_used[s_idx], :] = p_zi_x[s_idx]
            p_zi_x_ext[s_idx] = np.copy(p_zi_x_tmp)

        p_zi_yi = np.copy(p_zi_yi_ext)
        p_zi_x = np.copy(p_zi_x_ext)

        # update other output pdfs
        p_ziyix = np.tile(np.expand_dims(self.p_yix[sensor_idx], 0), np.append(self.Nz[sensor_idx], np.ones(2, dtype=int))) * np.tile(np.expand_dims(p_zi_yi[sensor_idx], -1), self.Nx)
        p_zi_x[sensor_idx] = (1 / np.tile(np.expand_dims(self.p_x, 0), np.append(self.Nz[sensor_idx], 1))) * np.sum(p_ziyix, 1)
        p_ziyix = None

        p_zx = np.ones(np.append(self.Nz, self.Nx))
        p_x_expand = np.tile(self.increase_dims(self.p_x, np.arange(0, self.Nsensors, dtype=int)), np.append(self.Nz, 1))
        for s_idx in range(0, self.Nsensors):
            j_elements = self.get_j_elements(s_idx)

            p_zi_x_expand = np.tile(self.increase_dims(p_zi_x[s_idx], j_elements), self.put_in_position(np.ones(self.Nsensors + 1, dtype=int), j_elements, np.array(self.Nz)[j_elements]))
            p_zx = p_zx * p_zi_x_expand
            p_zi_x_expand = None
        p_zx = p_zx * p_x_expand
        p_x_expand = None

        p_z = np.sum(p_zx, -1)
        with np.errstate(all='ignore'):
            p_x_z = p_zx / np.tile(np.expand_dims(p_z, -1), np.append(np.ones(self.Nsensors, dtype=int), self.Nx))
        p_x_z[np.isnan(p_x_z)] = 0

        return p_zi_yi[sensor_idx], p_zi_x[sensor_idx], p_zx, p_x_z, p_z


    # --------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------- helper functions ------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------

    def increase_dims(self, x_, dim_tuple_):
        # old_dimensions = x_.shape
        newx = np.copy(x_)
        dim_tuple_ = np.sort(dim_tuple_)
        for i in range(0, len(dim_tuple_)):
            newx = np.expand_dims(newx, dim_tuple_[i])
        return newx

    def put_in_position(self,x_,pos_,val_):
        newx = np.copy(x_)
        np.put(newx,pos_,val_)
        return newx

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

    def get_j_elements(self,sensor_idx):
        # get j-th elements -> all other sensors than the i-th sensor
        j_elements = np.arange(0, self.Nsensors, dtype=int)
        j_elements = np.delete(j_elements, sensor_idx)
        return j_elements

    def create_forward_channel_distribution(self, N_sensors, Nz, SNR):
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