import sys
import os
import numpy as np
import logging
import itertools


from generic_alternating_IB_class import GenericAIB
from iterative_ib_scalar_class import IterativeIBScalar
from sequential_distributed_iterative_ib_sideinformation_class import SequentialDistributedIterativeIBSideinformation


__author__ = "Steffen Steiner"
__copyright__ = "12.11.2018, Institute of Communications, University of Rostock "
__credits__ = ["Steffen Steiner"]
__version__ = "1.0"
__email__ = "steffen.steiner@uni-rostock.de"
__status__ = "Release"
__doc__="""This class can be used to perform the alternating Information Bottleneck algorithm"""


class SequentialDistributedIB(GenericAIB):
    """This class can be used to perform the alternating Information Bottleneck algorithm
        Args:
        input parameter
            p_xyi                   list of input joint pdfs for every sensor
            p_init                  list of initial distributions p_z_y for every sensor
            Nx                      cardinality of x
            Ny                      cardinality of y for all sensors
            Nz                      cardinality of z for all sensors
            beta_range              range of betas for every sensor
                min                 minimum for every sensor
                max                 max for every sensor

            params
                accuracy_aib            accuracy for aib method
                max_iter_aib            maximum iterations of aib method
                accuracy_bs             accuracy for bisection search
                max_iter_bs             maximum iterations of bisection search
                accuracy_ib             accuracy for ib method for every sensor
                max_iter_ib             maximum iterations of ib algorithm
                ib_solver               string of used ib algorithm
                                            -> 'iterative'
                sensor_opt_order        order of sensor optimization
                ib_conv_crit            convergence criterion of ib algorithm
                aib_conv_crit            convergence criterion of whole aib algorithm
    """

    # --------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------- initialization --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, Nsensors_, p_xyi_, p_init, Nx_, Ny_, Nz_, target_rates_, beta_range, params):
        GenericAIB.__init__(self, Nsensors_, None, p_xyi_, p_init, Nx_, Ny_, Nz_, target_rates_, beta_range, params)
        self.name = 'SIB_'

        #create one ib solver for with sideinformation and one scalar ib solver for every sensor
        solver_list_no_sideinfo_ = [[] for _ in range(0,self.Nsensors)]
        if self.params.get('ib_solver') == 'iterative':
            for idx in range(0, self.Nsensors):
                solver_list_no_sideinfo_[idx] = IterativeIBScalar(p_xyi_[idx], self.p_init[idx], self.Nx, self.Ny[idx], self.Nz[idx], None, self.params.get('accuracy_ib'), self.params.get('max_iter_ib'))
            self.solver_list_no_sideinfo = solver_list_no_sideinfo_
            self.solver_sideinfo = SequentialDistributedIterativeIBSideinformation(self.Nsensors, self.params.get('sensor_opt_order'), p_xyi_, self.p_init, self.Nx, self.Ny, self.Nz, None, self.params.get('accuracy_ib'),self.params.get('max_iter_ib'))

        #define output variables
        aib_result = {}
        aib_result['Iterations_aib'] = [[] for _ in range(0,self.params.get('max_iter_aib'))]
        aib_result['used_iter_aib'] = None
        aib_result['Init_sum_rate'] = None
        aib_result['H_x'] = -np.sum(self.p_x * np.log2(self.p_x))
        aib_result['I_xy'] = None
        aib_result['Final_Quantizers'] = [[] for _ in range(0,self.Nsensors)]
        aib_result['Final_p_zix'] = [[] for _ in range(0, self.Nsensors)]
        for aib_idx in range(0,self.params.get('max_iter_aib')):

            aib_result['Iterations_aib'][aib_idx] = {}
            aib_result['Iterations_aib'][aib_idx]['residual_aib'] = None
            aib_result['Iterations_aib'][aib_idx]['final_compression_rates'] = {}
            aib_result['Iterations_aib'][aib_idx]['final_compression_rates']['partial_sideinformation'] = None


            I__multi_xy = 0
            node_results = [[] for _ in range(0,self.Nsensors)]
            for node_idx in range(0, self.Nsensors):
                node_results[node_idx] = {}
                node_results[node_idx]['H_yi'] = -np.sum(self.p_yi[node_idx] * np.log2(self.p_yi[node_idx]))
                node_results[node_idx]['I_xyi'] = np.sum(self.p_yix[node_idx]* np.log2(self.p_yix[node_idx] \
                    /(np.tile(np.expand_dims(self.p_x,0),np.append(self.Ny[node_idx],1))*np.tile(np.expand_dims(self.p_yi[node_idx],-1),np.append(1,self.Nx)))))
                I__multi_xy = I__multi_xy + node_results[node_idx]['I_xyi']

                node_bs_iteration_results = [[] for _ in range(0,self.params.get('max_iter_bs'))]
                for bs_idx in range(0, self.params.get('max_iter_bs')):
                    node_bs_iteration_results[bs_idx] = {}
                    node_bs_iteration_results[bs_idx]['residual_ib'] = [None for _ in range(0,self.params.get('max_iter_ib'))]
                    node_bs_iteration_results[bs_idx]['used_iter_ib'] = None
                    node_bs_iteration_results[bs_idx]['beta'] = None
                    node_bs_iteration_results[bs_idx]['residual_bs'] = None
                    node_bs_iteration_results[bs_idx]['I_yizi'] = None
                    node_bs_iteration_results[bs_idx]['I_xzi'] = None
                    node_bs_iteration_results[bs_idx]['H_zi'] = None

                    node_bs_iteration_results[bs_idx]['I_yizi_zj'] = None
                    node_bs_iteration_results[bs_idx]['I_xzi_zj'] = None

                    node_bs_iteration_results[bs_idx]['I_xz'] = None
                    node_bs_iteration_results[bs_idx]['H_z'] = None
                    node_bs_iteration_results[bs_idx]['H_zi_zj'] = None



                    if self.params.get('store_inner_ib_results'):
                        node_bs_iteration_results[bs_idx]['I_yizi_zj_evolution'] = [None for _ in range(0, self.params.get('max_iter_ib'))]
                        node_bs_iteration_results[bs_idx]['I_xzi_zj_evolution'] = [None for _ in range(0, self.params.get('max_iter_ib'))]
                        node_bs_iteration_results[bs_idx]['I_yizi_evolution'] = [None for _ in range(0, self.params.get('max_iter_ib'))]
                        node_bs_iteration_results[bs_idx]['I_xzi_evolution'] = [None for _ in range(0, self.params.get('max_iter_ib'))]
                        node_bs_iteration_results[bs_idx]['H_zi_evolution'] = [None for _ in range(0, self.params.get('max_iter_ib'))]
                        node_bs_iteration_results[bs_idx]['I_xz_evolution'] = [None for _ in range(0, self.params.get('max_iter_ib'))]
                        node_bs_iteration_results[bs_idx]['H_z_evolution'] = [None for _ in range(0, self.params.get('max_iter_ib'))]
                        node_bs_iteration_results[bs_idx]['H_zi_zj_evolution'] = [None for _ in range(0, self.params.get('max_iter_ib'))]


                node_results[node_idx]['Iterations_bs'] = node_bs_iteration_results
                node_results[node_idx]['used_iter_bs'] = None
            aib_result['Iterations_aib'][aib_idx]['Node_results'] = node_results
            aib_result['I__multi_xy'] = I__multi_xy

        self.aib_result = aib_result



    # --------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------original sIB algorithm  ------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------
    def run(self):
        logging.debug('start simulation')
        self.run_aIB_algorithm()

    def run_aIB_algorithm(self):
        #calculate initial sum rate and system information
        self.determine_sum_rate(self.p_init,init=True)

        logging.debug('start aIB algorithm')

        iter_aib = 0
        residual_aib = 1
        p_zi_yi = np.copy(self.p_init)

        # repeat optimization until no significant changes occur anymore
        while ((residual_aib>self.params.get('accuracy_aib')) and (iter_aib < self.params.get('max_iter_aib'))):

            #safe previous p(z|y) for later residual calculation
            p_zi_yi_prev = np.copy(p_zi_yi)

            # calculate p_zi_x for all sensors  -> later just update the specific p_zi_x of sensor i
            p_zi_x = self.determine_p_zi_x(p_zi_yi)


            #check for inverse optimization
            if self.params.get('inverse_optimization'):
                opt_order = np.copy(self.params.get('sensor_opt_order')[::-1])
            else:
                opt_order = np.copy(self.params.get('sensor_opt_order'))
            # loop over all sensors to update individual quantizers
            for sensor_idx in opt_order:

                logging.debug('optimize sensor: '+str(sensor_idx))

                residual_bs = 1
                iter_bs = 0
                lambda_bounds = [self.beta_range.get('min')[sensor_idx],self.beta_range.get('max')[sensor_idx]]

                # loop for adjusting beta -> bisection search
                while ((residual_bs > self.params.get('accuracy_bs')) and (iter_bs < self.params.get('max_iter_bs'))):

                    beta = np.inf
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['beta'] = beta

                    #-----------------------iterative ib algorithm---------------------------------------------------

                    # always initialize the current sensor as the initial quantizer
                    p_zi_yi[sensor_idx] = np.copy(self.p_init[sensor_idx])
                    p_zi_x[sensor_idx] = self.determine_p_zi_x(p_zi_yi,sensor_idx)

                    iter_ib = 0;
                    residual_ib = 1;

                    while ((residual_ib > self.params.get('accuracy_ib')) and (iter_ib < self.params.get('max_iter_ib'))):

                        p_zi_yi_prev_ib = np.copy(p_zi_yi[sensor_idx])

                        if iter_ib%10 == 0:
                            logging.info('SIB::'+'\t'
                                +'Permutation: ' + str(self.params.get('sensor_opt_order')) + ' iterIB: ' + str(iter_ib) + ' iterBS: ' + str(iter_bs) + ' node: ' + str(sensor_idx) + ' iterSIB: ' + str(
                                    iter_aib)
                                + '\t' + 'optimized with side-information: ' + str(bool(self.params.get('sensor_opt_with_side')[sensor_idx])))

                        if self.params.get('sensor_opt_with_side')[sensor_idx]:
                            p_zi_yi[sensor_idx], p_zi_x[sensor_idx], p_zx, p_x_z, p_z = self.solver_sideinfo.run_one_iteration(sensor_idx, p_zi_yi, p_zi_x, beta, iter_ib, 'partial')

                        else:
                            p_zi_yi[sensor_idx], p_zi_x[sensor_idx] = self.solver_list_no_sideinfo[sensor_idx].run_one_iteration(p_zi_yi[sensor_idx], beta)

                        if self.params.get('store_inner_ib_results'):
                            if self.params.get('sensor_opt_with_side')[sensor_idx]:
                                self.determine_mutual_information(p_zi_yi, p_zi_x, p_zx, p_x_z, p_z, sensor_idx, iter_aib, iter_bs,iter_ib=iter_ib)
                            else:
                                self.determine_mutual_information(p_zi_yi, p_zi_x, p_zx=None, p_x_z=None, p_z=None, sensor_idx=sensor_idx, iter_aib=iter_aib, iter_bs=iter_bs,scalar=True,iter_ib=iter_ib)

                        #calculate residual
                        residual_ib = self.calculate_residual_IB(p_zi_yi[sensor_idx],p_zi_yi_prev_ib,self.params.get('ib_conv_crit'),sensor_idx)

                        self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['residual_ib'][iter_ib] = residual_ib
                        iter_ib += 1
                    # ------------------------------------------------------------------------------------------------

                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['used_iter_ib'] = iter_ib

                    # determine mutual information and calculate accuracy of bisection search and adjusting beta
                    if self.params.get('sensor_opt_with_side')[sensor_idx]:
                        self.determine_mutual_information(p_zi_yi, p_zi_x, p_zx, p_x_z, p_z, sensor_idx, iter_aib, iter_bs)

                        if sensor_idx != self.params.get('sensor_opt_order')[0]:
                            residual_bs, lambda_bounds = self.calculate_residual_BS(self.target_rates[sensor_idx], self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_zj'],False, lambda_bounds)
                        else:
                            residual_bs, lambda_bounds = self.calculate_residual_BS(self.target_rates[sensor_idx], self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi'],False,lambda_bounds)
                    else:
                        self.determine_mutual_information(p_zi_yi, p_zi_x, p_zx = None, p_x_z= None, p_z = None, sensor_idx = sensor_idx, iter_aib= iter_aib, iter_bs = iter_bs,scalar=True)
                        residual_bs, lambda_bounds = self.calculate_residual_BS(self.target_rates[sensor_idx], self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi'],True,lambda_bounds)
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['residual_bs'] = residual_bs

                    iter_bs += 1

                self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['used_iter_bs'] = iter_bs

                logging.debug('optimize sensor: ' + str(sensor_idx) + ':: done')

            logging.debug('aib iteration: '+str(iter_aib)+':: done')

            #calculate residual for aIB algorithm
            residual_aib = self.calculate_residual_aIB(p_zi_yi, p_zi_yi_prev, self.params.get('aib_conv_crit'))
            self.aib_result['Iterations_aib'][iter_aib]['residual_aib'] = residual_aib


            self.determine_compression_rates(p_zi_yi, p_zi_x,iter_aib)

            iter_aib += 1

        self.aib_result['used_iter_aib'] = iter_aib
        self.aib_result['Final_Quantizers'] = p_zi_yi
        self.aib_result['Final_p_zix'] = p_zi_x


    # --------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------functions to test the sIB algorithm  -----------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------
    #...
    #---------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------- methods for evaluation ------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------

    def determine_p_zi_x(self,p_zi_yi,sensor_idx = None):
        """ Determine p_zi_x for all sensors """

        # p(z|x) = (1/p(x))* sum(p(z|y)*p(x,y), over y)
        p_zi_x = [[] for _ in range(0, self.Nsensors)]
        for idx in range(0, self.Nsensors):
            p_ziyixi = np.tile(np.expand_dims(self.p_yix[idx], 0), np.append(self.Nz[idx], np.ones(2, dtype=int))) * np.tile(np.expand_dims(p_zi_yi[idx], -1), self.Nx)
            p_zi_x[idx] = (1 / np.tile(np.expand_dims(self.p_x, 0), np.append(self.Nz[idx], 1))) * np.sum(p_ziyixi, 1)
        if sensor_idx == None:
            return p_zi_x
        else:
            return p_zi_x[sensor_idx]

    def determine_sum_rate(self,p_zi_yi, init = False, p_zi_x = None, sensor_idx = None, iter_aib = None, iter_bs=None, p_zx = None, p_x_z=None, iter_ib = None, scalar = False):
        """ Determine the relevant mutual information and save them in the aib_result data structure for multi sensors"""

        logging.debug('calculate initial sum rate and entropy of Z')

        if init:
            #IB has not been running yet -> use function to determine p(zi|x)
            p_zi_x = self.determine_p_zi_x(p_zi_yi)
            
            for idx in range(0, self.Nsensors):
                p_ziyixi = np.tile(np.expand_dims(self.p_yix[idx],0),np.append(self.Nz[idx],np.ones(2,dtype=int))) * np.tile(np.expand_dims(p_zi_yi[idx],-1),self.Nx)
                p_zi_x[idx] = (1/np.tile(np.expand_dims(self.p_x,0),np.append(self.Nz[idx],1))) * np.sum(p_ziyixi,1)

            p_zx = np.ones(np.append(self.Nz, self.Nx))
            p_x_expand = np.tile(self.increase_dims(self.p_x, np.arange(0, self.Nsensors, dtype=int)), np.append(self.Nz, 1))
            for s_idx in range(0,self.Nsensors):
                # get j-th elements -> all other sensors than the i-th sensor
                j_elements = np.arange(0, self.Nsensors, dtype=int)
                j_elements = np.delete(j_elements, s_idx)

                p_zi_x_expand = np.tile(self.increase_dims(p_zi_x[s_idx],j_elements),self.put_in_position(np.ones(self.Nsensors+1,dtype=int),j_elements,np.array(self.Nz)[j_elements]))
                p_zx = p_zx * p_zi_x_expand
                p_zi_x_expand = None
            p_zx = p_zx * p_x_expand
            p_x_expand = None

            p_z = np.sum(p_zx,-1)
            p_x_z = p_zx / np.tile(np.expand_dims(p_z, -1), np.append(np.ones(self.Nsensors, dtype=int), self.Nx))

        # determine I(X;Z)
        with np.errstate(all='ignore'):
            tmp1 = np.log2(self.p_x)
            tmp1[np.isnan(tmp1)] = 0  # remove NaN
            tmp2 = np.log2(p_x_z)
            tmp2[np.isnan(tmp2)] = 0  # remove NaN
            tmp = p_zx * (tmp2 - np.tile(tmp1, np.append(self.Nz, 1)))
            tmp[np.isnan(tmp)] = 0  # remove NaN
        if init:
            if self.aib_result['Init_sum_rate']== None:
                self.aib_result['Init_sum_rate'] = np.sum(tmp)
            if scalar:
                if iter_ib == None:
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_xz'] = np.sum(tmp)
                else:
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_xz_evolution'][iter_ib] = np.sum(tmp)
        else:
            if iter_ib == None:
                self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_xz'] = np.sum(tmp)
            else:
                self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_xz_evolution'][iter_ib] = np.sum(tmp)
        tmp = None
        tmp1 = None
        tmp2 = None

        logging.debug('calculating sum rate and entropy of Z:: done')


    def determine_mutual_information(self,p_zi_yi, p_zi_x, p_zx, p_x_z, p_z, sensor_idx, iter_aib, iter_bs,scalar = False, iter_ib = None):
        """ Determine the mutual informations and save them in the aib_result data structure"""

        logging.debug('calculate mutual informations')

        # determine I(Yi;Zi)
        p_ziyi = p_zi_yi[sensor_idx] * np.tile(np.expand_dims(self.p_yi[sensor_idx],0),np.append(self.Nz[sensor_idx],1))
        p_zi = np.sum(p_ziyi,1)
        with np.errstate(all='ignore'):
            tmp1 = np.log2(p_zi)
            tmp1[np.isnan(tmp1)] = 0  # remove NaN
            tmp2 = np.log2(p_zi_yi[sensor_idx])
            tmp2[np.isnan(tmp2)] = 0  # remove NaN
            tmp = p_ziyi * (tmp2 - np.tile(np.expand_dims(tmp1, -1), np.append(1, self.Ny[sensor_idx])))
            tmp[np.isnan(tmp)] = 0  # remove NaN
        if iter_ib == None:
            self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi'] = np.sum(tmp)
        else:
            self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_evolution'][iter_ib] = np.sum(tmp)
        tmp = None
        tmp1 = None
        tmp2 = None

        # determine I(X;Zi)
        p_zix = p_zi_x[sensor_idx] * np.tile(np.expand_dims(self.p_x,0),np.append(self.Nz[sensor_idx],1))
        with np.errstate(all='ignore'):
            tmp1 = np.log2(p_zi)
            tmp1[np.isnan(tmp1)] = 0  # remove NaN
            tmp2 = np.log2(p_zi_x[sensor_idx])
            tmp2[np.isnan(tmp2)] = 0  # remove NaN
            tmp = p_zix * (tmp2 - np.tile(np.expand_dims(tmp1,-1), np.append(1,self.Nx)))
            tmp[np.isnan(tmp)] = 0  # remove NaN
        if iter_ib == None:
            self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_xzi'] = np.sum(tmp)
        else:
            self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_xzi_evolution'][iter_ib] = np.sum(tmp)
        tmp = None
        tmp1 = None
        tmp2 = None

        # determine H(Zi)
        with np.errstate(all='ignore'):
            tmp = np.log2(p_zi) * p_zi
            tmp[np.isnan(tmp)] = 0  # remove NaN
        if iter_ib == None:
            self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_zi'] = - np.sum(tmp)
        else:
            self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_zi_evolution'][iter_ib] = - np.sum(tmp)
        tmp = None

        if scalar or (self.params.get('chain_optimization') and self.params.get('sensor_opt_order')[0]==sensor_idx):
            # determine I(X;Z)
            if iter_ib == None:
                self.determine_sum_rate(p_zi_yi, init=True, sensor_idx=sensor_idx, iter_aib=iter_aib, iter_bs=iter_bs, scalar = True)
            else:
                self.determine_sum_rate(p_zi_yi, init=True, sensor_idx=sensor_idx, iter_aib=iter_aib, iter_bs=iter_bs, iter_ib = iter_ib, scalar = True)

            # store compression rate
            if iter_ib == None:
                self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_zj'] \
                    = self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi']
            else:
                self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_zj_evolution'][iter_ib] \
                    = self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_evolution'][iter_ib]

        else:
            # determine H(Z)
            with np.errstate(all='ignore'):
                tmp = np.log2(p_z) * p_z
                tmp[np.isnan(tmp)] = 0  # remove NaN
            if iter_ib == None:
                self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_z'] = - np.sum(tmp)
            else:
                self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_z_evolution'][iter_ib] = - np.sum(tmp)
            tmp = None

            # determine I(X;Z)
            if iter_ib == None:
                self.determine_sum_rate(p_zi_yi, init=False, p_zi_x=p_zi_x, sensor_idx=sensor_idx, iter_aib=iter_aib, iter_bs=iter_bs, p_zx=p_zx, p_x_z=p_x_z)
            else:
                self.determine_sum_rate(p_zi_yi, init=False, p_zi_x=p_zi_x, sensor_idx=sensor_idx, iter_aib=iter_aib, iter_bs=iter_bs, p_zx=p_zx, p_x_z=p_x_z, iter_ib = iter_ib)

            # determine I(Yi;Zi|Zj~=i) -> mutual information between Yi and Zi given Zj~=i
            if sensor_idx == self.params.get('sensor_opt_order')[0]:
                if iter_ib == None:
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_zj'] \
                        = self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi']
                else:
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_zj_evolution'][iter_ib] \
                        = self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_evolution'][iter_ib]
            else:
                sensor_list_pos = np.where(np.array(self.params.get('sensor_opt_order'), dtype=int) == sensor_idx)[0][0]
                partial_elements = self.params.get('sensor_opt_order')[:sensor_list_pos + 1]
                partial_j_elements = self.params.get('sensor_opt_order')[:sensor_list_pos]
                diff_node = np.setdiff1d(partial_elements, partial_j_elements)[0]
                diff_pos = self.dimension_difference(np.sort(partial_elements), np.sort(partial_j_elements))[0][0]
                remove_elements = self.params.get('sensor_opt_order')[sensor_list_pos::]
                p_z_partial = np.sum(p_z, tuple(remove_elements[1::]))
                p_zj_partial = np.sum(p_z, tuple(remove_elements))
                p_zi_zj_partial = p_z_partial / np.tile(self.increase_dims(p_zj_partial, [diff_pos]), self.put_in_position(np.ones(len(partial_elements), dtype=int), diff_pos, self.Nz[diff_node]))


                with np.errstate(all='ignore'):
                    tmp = np.log2(p_zi_zj_partial) * p_z_partial
                    tmp[np.isnan(tmp)] = 0  # remove NaN
                if iter_ib == None:
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_zi_zj'] = - np.sum(tmp)
                else:
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_zi_zj_evolution'][iter_ib] = - np.sum(tmp)
                tmp = None


                if iter_ib == None:
                    I_yizi_zj = self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi'] \
                                - self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_zi'] \
                                + self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_zi_zj']
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_zj'] = I_yizi_zj
                else:
                    I_yizi_zj = self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_evolution'][iter_ib] \
                                - self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_zi_evolution'][iter_ib] \
                                + self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['H_zi_zj_evolution'][iter_ib]
                    self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_yizi_zj_evolution'][iter_ib] = I_yizi_zj

            # determine I(X;Zi|Zj~=i) -> mutual information between X and Zi given Zj~=i)
            p_zjx = np.sum(p_zx, sensor_idx)
            p_zj = np.sum(p_zjx,-1)
            with np.errstate(all='ignore'):
                p_x_zj = p_zjx / np.tile(np.expand_dims(p_zj, -1), np.append(np.ones(self.Nsensors - 1, dtype=int), self.Nx))
                p_x_zj[np.isnan(p_x_zj)] = 0  # remove NaN
                tmp1 = np.log2(p_x_zj)
                tmp1[np.isnan(tmp1)] = 0  # remove NaN
                tmp2 = np.log2(p_x_z)
                tmp2[np.isnan(tmp2)] = 0  # remove NaN
                tmp = p_zx * (tmp2 - np.tile(np.expand_dims(tmp1, sensor_idx), self.put_in_position(np.ones(self.Nsensors + 1, dtype=int), sensor_idx, self.Nz[sensor_idx])))
                tmp[np.isnan(tmp)] = 0  # remove NaN
            if iter_ib == None:
                self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_xzi_zj'] = np.sum(tmp)
            else:
                self.aib_result['Iterations_aib'][iter_aib]['Node_results'][sensor_idx]['Iterations_bs'][iter_bs]['I_xzi_zj_evolution'][iter_ib] = np.sum(tmp)
            tmp = None
            tmp1 = None
            tmp2 = None

        logging.debug('calculate mutual informations:: done')


    def determine_vector_compression_rate(self,p_zi_yi, p_zi_x):
        """ Determine the mutual information between vector z and vector y -> not used atm"""

        p_zx = np.ones(np.append(self.Nz, self.Nx))
        p_x_expand = np.tile(self.increase_dims(self.p_x, np.arange(0, self.Nsensors, dtype=int)), np.append(self.Nz, 1))
        for s_idx in range(0, self.Nsensors):
            # get j-th elements -> all other sensors than the i-th sensor
            j_elements = np.arange(0, self.Nsensors, dtype=int)
            j_elements = np.delete(j_elements, s_idx)

            p_zi_x_expand = np.tile(self.increase_dims(p_zi_x[s_idx], j_elements), self.put_in_position(np.ones(self.Nsensors + 1, dtype=int), j_elements, np.array(self.Nz)[j_elements]))
            p_zx = p_zx * p_zi_x_expand
            p_zi_x_expand = None
        p_zx = p_zx * p_x_expand
        p_x_expand = None
        p_z = np.sum(p_zx, -1)

        p_ziyi_vec = [[] for _ in range(0, self.Nsensors)]
        for sensor_idx in range(0, self.Nsensors):
            p_ziyi_vec[sensor_idx] = p_zi_yi[sensor_idx] * np.tile(np.expand_dims(self.p_yi[sensor_idx], 0), np.append(self.Nz[sensor_idx], 1))

        # sumrate: I(Y;Z) = H(Z)- sum(H(zi,yi), over i)
        with np.errstate(all='ignore'):
            tmp = np.log2(p_z) * p_z
            tmp[np.isnan(tmp)] = 0  # remove NaN
        H_z = - np.sum(tmp)
        tmp = None

        H_zi_yi_vec = [[] for _ in range(0, self.Nsensors)]
        for s_idx in range(0, self.Nsensors):
            with np.errstate(all='ignore'):
                tmp = np.log2(p_zi_yi[s_idx]) * p_ziyi_vec[s_idx]
                tmp[np.isnan(tmp)] = 0  # remove NaN
            H_zi_yi_vec[s_idx] = - np.sum(tmp)
            tmp = None
            tmp2 = None
        I_yz = H_z - np.sum(H_zi_yi_vec)
        return I_yz

    def determine_compression_rates(self,p_zi_yi, p_zi_x, iter_aib):
        """ Determine compression rates for all sensors
        -> just used to calculate final compression rates after every sensor was optimized"""

        logging.debug('calculate compression rates')
        comp_rates_partial = [[] for _ in range(0, self.Nsensors)]

        p_zx = np.ones(np.append(self.Nz, self.Nx))
        p_x_expand = np.tile(self.increase_dims(self.p_x, np.arange(0, self.Nsensors, dtype=int)), np.append(self.Nz, 1))
        for s_idx in range(0, self.Nsensors):
            # get j-th elements -> all other sensors than the i-th sensor
            j_elements = np.arange(0, self.Nsensors, dtype=int)
            j_elements = np.delete(j_elements, s_idx)

            p_zi_x_expand = np.tile(self.increase_dims(p_zi_x[s_idx], j_elements), self.put_in_position(np.ones(self.Nsensors + 1, dtype=int), j_elements, np.array(self.Nz)[j_elements]))
            p_zx = p_zx * p_zi_x_expand
            p_zi_x_expand = None
        p_zx = p_zx * p_x_expand
        p_x_expand = None
        p_z = np.sum(p_zx, -1)

        p_ziyi_vec = [[] for _ in range(0, self.Nsensors)]

        for sensor_idx in range(0,self.Nsensors):

            # determine I(Yi;Zi)
            p_ziyi_vec[sensor_idx] = p_zi_yi[sensor_idx] * np.tile(np.expand_dims(self.p_yi[sensor_idx], 0), np.append(self.Nz[sensor_idx], 1))
            p_zi = np.sum(p_ziyi_vec[sensor_idx], 1)
            with np.errstate(all='ignore'):
                tmp1 = np.log2(p_zi)
                tmp1[np.isnan(tmp1)] = 0  # remove NaN
                tmp2 = np.log2(p_zi_yi[sensor_idx])
                tmp2[np.isnan(tmp2)] = 0  # remove NaN
                tmp = p_ziyi_vec[sensor_idx] * (tmp2 - np.tile(np.expand_dims(tmp1, -1), np.append(1, self.Ny[sensor_idx])))
                tmp[np.isnan(tmp)] = 0  # remove NaN
            I_yizi = np.sum(tmp)
            tmp = None
            tmp1 = None
            tmp2 = None

            # determine H(Zi)
            with np.errstate(all='ignore'):
                tmp = np.log2(p_zi) * p_zi
                tmp[np.isnan(tmp)] = 0  # remove NaN
            H_zi = - np.sum(tmp)
            tmp = None


            if sensor_idx == self.params.get('sensor_opt_order')[0]:
                comp_rates_partial[sensor_idx] = I_yizi
            else:
                sensor_list_pos = np.where(np.array(self.params.get('sensor_opt_order'), dtype=int) == sensor_idx)[0][0]
                partial_elements = self.params.get('sensor_opt_order')[:sensor_list_pos + 1]
                partial_j_elements = self.params.get('sensor_opt_order')[:sensor_list_pos]
                diff_node = np.setdiff1d(partial_elements, partial_j_elements)[0]
                diff_pos = self.dimension_difference(np.sort(partial_elements), np.sort(partial_j_elements))[0][0]
                remove_elements = self.params.get('sensor_opt_order')[sensor_list_pos::]
                p_z_partial = np.sum(p_z, tuple(remove_elements[1::]))
                p_zj_partial = np.sum(p_z, tuple(remove_elements))
                p_zi_zj_partial = p_z_partial / np.tile(self.increase_dims(p_zj_partial, [diff_pos]), self.put_in_position(np.ones(len(partial_elements), dtype=int), diff_pos, self.Nz[diff_node]))

                with np.errstate(all='ignore'):
                    tmp = np.log2(p_zi_zj_partial) * p_z_partial
                    tmp[np.isnan(tmp)] = 0  # remove NaN
                H_zi_zj = - np.sum(tmp)
                tmp = None

                comp_rates_partial[sensor_idx] = I_yizi - H_zi + H_zi_zj

        self.aib_result['Iterations_aib'][iter_aib]['final_compression_rates']['partial_sideinformation'] = comp_rates_partial


        # sumrate: I(Y;Z) = H(Z)- sum(H(zi,yi), over i)
        with np.errstate(all='ignore'):
            tmp = np.log2(p_z) * p_z
            tmp[np.isnan(tmp)] = 0  # remove NaN
        H_z = - np.sum(tmp)
        tmp = None


        H_zi_yi_vec = [[] for _ in range(0, self.Nsensors)]
        for s_idx in range(0, self.Nsensors):
            with np.errstate(all='ignore'):
                tmp = np.log2(p_zi_yi[s_idx]) * p_ziyi_vec[s_idx]
                tmp[np.isnan(tmp)] = 0  # remove NaN
            H_zi_yi_vec[s_idx] = - np.sum(tmp)
            tmp = None
            tmp2 = None
        I_yz = H_z - np.sum(H_zi_yi_vec)

        # calculate additional conditions
        sensor_list = range(0, sensor_idx + 1)
        powerset = [x for length in range(len(sensor_list) + 1) for x in itertools.combinations(sensor_list, length)]
        del powerset[0]
        conditions = [[] for _ in range(0, len(powerset))]

        # last object
        conditions[-1] = {}
        conditions[-1]['variable'] = powerset[-1]
        conditions[-1]['condition'] = ()
        conditions[-1]['I(Y_variable;Z_variable|Z_condition)'] = I_yz
        del powerset[-1]

        variable_idx = len(powerset) - 1
        for condition_idx in range(0, len(powerset)):
            conditions[condition_idx] = {}
            conditions[condition_idx]['variable'] = powerset[variable_idx]
            conditions[condition_idx]['condition'] = powerset[condition_idx]

            # fill in already calculated conditions
            if len(conditions[condition_idx]['variable']) == 1:
                sensor = conditions[condition_idx]['variable'][0]

                # determine H(Zi|Zj)
                p_zj = np.sum(p_z, sensor)
                p_zi_zj = p_z / np.tile(np.expand_dims(p_zj, sensor), self.put_in_position(np.ones(self.Nsensors, dtype=int), sensor, self.Nz[sensor]))
                with np.errstate(all='ignore'):
                    tmp = np.log2(p_zi_zj) * p_z
                    tmp[np.isnan(tmp)] = 0  # remove NaN
                H_zi_zj = - np.sum(tmp)
                tmp = None
                conditions[condition_idx]['I(Y_variable;Z_variable|Z_condition)'] = I_yizi - H_zi + H_zi_zj

            else:
                # calculate H(Z_variable|Z_condition)
                variable = powerset[variable_idx]
                condition = powerset[condition_idx]
                p_zcond = np.sum(p_z, variable)
                p_zvar_zcond = p_z / np.tile(self.increase_dims(p_zcond, variable), self.put_in_position(np.ones(self.Nsensors, dtype=int), variable, np.array(self.Nz)[tuple([variable])]))
                with np.errstate(all='ignore'):
                    tmp = np.log2(p_zvar_zcond) * p_z
                    tmp[np.isnan(tmp)] = 0  # remove NaN
                H_zvar_zcond = - np.sum(tmp)

                H_sum = 0
                for s_idx in range(0, self.Nsensors):
                    if s_idx in variable:
                        H_sum = H_sum + H_zi_yi_vec[s_idx]
                conditions[condition_idx]['I(Y_variable;Z_variable|Z_condition)'] = H_zvar_zcond - H_sum
            variable_idx -= 1

        self.aib_result['Iterations_aib'][iter_aib]['all_rate_conditions'] = conditions


    def calculate_residual_IB(self,Q,Q_prev,method,sensor_idx):
    # def calculate_residual_IB(self, Q, Q_prev, method, sensor_idx, beta, iter_aib, iter_bs, iter_ib):
        """ Return residual of the IB algorithm given a quantizer Q and the previous quantizer Q_prev"""
        if method == 'MAE':
            # calculate the MAE between previous and current quantizer statistics
            residual_ib = np.sum(np.abs(Q.flatten() - Q_prev.flatten())) / (self.Ny[sensor_idx] * self.Nz[sensor_idx])
        elif method == 'MSE':
            # calculate the MSE between previous and current quantizer statistics
            residual_ib = np.sum(np.power(Q.flatten() - Q_prev.flatten(), 2)) / (self.Ny[sensor_idx] * self.Nz[sensor_idx])
        elif method == 'KL':
            # calculate the KL-divergence between previous and current quantizer statistics
            residual_ib = np.sum(self.res_KLD(Q, Q_prev) * self.p_yi[sensor_idx])
        elif method == 'JS':
            # calculate the JS-divergence between previous and current quantizer statistics
            residual_ib = np.sum(self.res_JSD(Q, Q_prev, [0.5, 0.5]) * self.p_yi[sensor_idx])

        else:
            raise RuntimeError('ib convergence criterion unknown!');
        return residual_ib


    def calculate_residual_BS(self,target_rate,current_rate,scalar,lambda_bounds=None):
        """ Return accuracy of bisection search and new beta interval"""
        residual_bs = np.abs(target_rate - current_rate)
        # adjusting beta
        if lambda_bounds!= None:
            if not scalar:
                # if current_rate < target_rate: for 1/beta
                if current_rate < target_rate:
                    # modify upper bound
                    lambda_bounds[1] = np.mean(lambda_bounds)
                else:
                    # modify lower bound
                    lambda_bounds[0] = np.mean(lambda_bounds)
            else:
                # if current_rate > target_rate:    for beta
                if current_rate > target_rate:
                    # modify upper bound
                    lambda_bounds[1] = np.mean(lambda_bounds)
                else:
                    # modify lower bound
                    lambda_bounds[0] = np.mean(lambda_bounds)
        return residual_bs, lambda_bounds

    def calculate_residual_aIB(self,Q_all,Q_all_prev,method):
        """ Return residual of the IB algorithm given a quantizer Q and the previous quantizer Q_prev"""
        if method == 'KL':
            # calculate the KL-divergence between previous and current quantizer statistics
            residual_aib = 0
            for sensor_idx in range(0, self.Nsensors):
                # tmp = self.res_KLD(Q_all[sensor_idx],Q_all_prev[sensor_idx])* np.tile(self.p_yi[sensor_idx], np.append(self.Nz[sensor_idx], 1))
                tmp = self.res_KLD(Q_all[sensor_idx], Q_all_prev[sensor_idx]) * self.p_yi[sensor_idx]
                residual_aib = residual_aib + np.sum(tmp)
            residual_aib = residual_aib / self.Nsensors
        elif method == 'JS':
            # calculate the JS-divergence between previous and current quantizer statistics
            residual_aib = 0
            for sensor_idx in range(0, self.Nsensors):
                # tmp = self.res_JSD(Q_all[sensor_idx], Q_all_prev[sensor_idx],[0.5,0.5]) * np.tile(self.p_yi[sensor_idx], np.append(self.Nz[sensor_idx], 1))
                tmp = self.res_JSD(Q_all[sensor_idx], Q_all_prev[sensor_idx],[0.5,0.5]) * self.p_yi[sensor_idx]
                residual_aib = residual_aib + np.sum(tmp)
            residual_aib = residual_aib / self.Nsensors
        return residual_aib

    def res_KLD(self,Q1,Q2):
        """ Return a vector for y of the KL Divergence for two input quantizers"""
        with np.errstate(all='ignore'):
            tmp = np.log2(Q1) - np.log2(Q2)
            tmp[np.isnan(tmp)] = 0
            tmp[np.isinf(tmp)] = 0
            KL = np.sum(Q1*tmp,axis=0)
            KL[np.isnan(KL)] = 0  # remove NaN
        return KL

    def res_JSD(self,Q1, Q2, pi_weigthts):
        """ Return a vector for y of the JS Divergence for two input quantizers"""
        p_bar = pi_weigthts[0] * Q1 + pi_weigthts[1] * Q2
        KL1 = self.res_KLD(Q1,p_bar)
        KL2 = self.res_KLD(Q2,p_bar)
        return pi_weigthts[0] * KL1 + pi_weigthts[1] * KL2

    def get_results(self):
        return self.aib_result

    # --------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------- helper functions ------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------

    def increase_dims(self, x_, dim_tuple_):
        newx = np.copy(x_)
        dim_tuple_ = np.sort(dim_tuple_)
        for i in range(0, len(dim_tuple_)):
            newx = np.expand_dims(newx, dim_tuple_[i])
        return newx

    def put_in_position(self,x_,pos_,val_):
        newx = np.copy(x_)
        np.put(newx,pos_,val_)
        return newx

    def dimension_difference(self, shape1, shape2):
        if len(shape1) > len(shape2):
            diff_elements = np.setdiff1d(shape1, shape2)
            diff_pos = np.where(shape1 == diff_elements)
        elif len(shape1) < len(shape2):
            diff_elements = np.setdiff1d(shape2, shape1)
            diff_pos = np.where(shape2 == diff_elements)
        else:
            difference = []
            for idx in range(0, len(shape1)):
                if shape1[idx] != shape2[idx]:
                    difference = difference + [idx]
            diff_pos = difference
        return diff_pos


