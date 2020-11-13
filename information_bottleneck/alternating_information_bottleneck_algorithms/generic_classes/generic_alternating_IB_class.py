import sys
import os
import numpy as np

__author__ = "Steffen Steiner"
__copyright__ = "14.11.2018, Institute of Communications, University of rostock"
__credits__ = ["Steffen Steiner"]
__version__ = "1.0"
__email__ = "steffen.steiner@uni-rostock.de"
__status__ = "Release"
__doc__ = """This module contains the definition of the abstract alternating information bottleneck class.
          """


class GenericAIB:
    """Common base class for all alternating Information Bottleneck classes
    Args:
    input parameter
        p_xy                    joint distribution of x and all y
        p_xyi                   list of input joint pdfs for every sensor
        p_init                  list of initial distributions p_z_y for every sensor
        Nx                      cardinality of x
        Ny                      cardinality of y for all sensors
        Nz                      cardinality of z for all sensors
        init_beta               list of initialized betas for every sensor
        accuracy                list of accuracies for every sensor
        max_iter_ib                list of maximum iterations of ib algorithm

        ib_solver               string of used ib algorithm
                                    -> 'iterative'
        beta_opt_strategy       strategy for optimizing beta
                                    -> 'successive'
        sensor_opt_order        order of sensor optimization
                                    -> 'BFS'

    """

    def __init__(self, Nsensors_, p_xy_, p_xyi_, p_init_, Nx_, Ny_, Nz_, target_rates_, beta_range_, params_):
        # initialize parameters
        self.Nsensors = Nsensors_
        self.p_x = np.sum(p_xyi_[0],1)
        self.p_yi = [np.sum(x, 0) for x in p_xyi_]
        self.p_xy = p_xy_
        self.p_init = p_init_
        self.Nz = Nz_
        self.Nx = Nx_
        self.Ny = Ny_
        self.target_rates = target_rates_
        self.beta_range = beta_range_
        self.params = params_

        self.p_yix = [[] for _ in range(0, self.Nsensors)]
        # calculate p_yix for all sensors
        for s_idx in range(0, self.Nsensors):
            self.p_yix[s_idx] = np.transpose(p_xyi_[s_idx])

        # initialize unused parameters
        self.name = 'GenericAIB'

    def run(self):
        """only template that will be used by the specific implementations later."""
        pass



