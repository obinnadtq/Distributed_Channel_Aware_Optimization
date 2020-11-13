import sys
import os
import numpy as np

__author__ = "Steffen Steiner"
__copyright__ = "14.11.2018, Institute of Communications, University of rostock"
__credits__ = ["Steffen Steiner"]
__version__ = "1.0"
__email__ = "steffen.steiner@uni-rostock.de"
__status__ = "Release"
__doc__ = """This module contains the definition of the abstract information bottleneck class for multiple sensors.
          """


class GenericIBSideinformation:
    """Common base class for all Information Bottleneck solver classes for multiple sensors
        Args:
    input parameter
        Nsensors                number of sensors
        p_xy                    input joint pdf for x and all y
        p_xyi
        p_init                  initial pmf p_z_y for all sensors
        Nx                      cardinality of x
        Ny                      cardinality of y for all sensors
        Nz                      cardinality of z for all sensors
        beta                    beta for all sensors
        accuracy                accuracy for all sensors
        max_iter                maximum number of iterations of ib algorithm
    mutual information
        MI_XZ                   mutual information of output I(X;Z)
        MI_XY                   mutual information of input I(X;Y)
    output PDF_s
        p_z_y
        p_x_z
        p_z
    """

    def __init__(self,Nsensors_, p_xy_, p_xyi_, p_init_, Nx_, Ny_, Nz_, beta_, accuracy_ , max_iter_):
        # initialize parameters
        self.Nsensors = Nsensors_
        self.p_xy = p_xy_
        # self.p_xyi = p_xyi_
        self.p_x = np.sum(p_xyi_[0],1)
        # self.p_x = np.sum(np.copy(p_xy_), tuple(np.arange(1, len(Ny_) + 1, dtype=int)))
        self.p_init = p_init_
        self.Nx = Nx_
        self.Ny = Ny_
        self.Nz = Nz_
        self.beta = beta_
        self.accuracy = accuracy_
        self.max_iter = max_iter_

        # initialize unused parameters
        self.p_z_y = p_init_
        self.name = 'GenericIBSideinformationClass'

        self.p_yix = [[] for _ in range(0, self.Nsensors)]
        # calculate p_yix for all sensors
        for s_idx in range(0, self.Nsensors):
            self.p_yix[s_idx] = np.transpose(p_xyi_[s_idx])

        self.p_yi_x = [[] for _ in range(0, self.Nsensors)]
        # calculate p_yi_x for all sensors
        for s_idx in range(0, self.Nsensors):
            self.p_yi_x[s_idx] = self.p_yix[s_idx] / np.tile(self.p_x,(self.Ny[s_idx],1))

        self.p_yi = [[] for _ in range(0, self.Nsensors)]
        # calculate p_yi for all sensors
        for s_idx in range(0, self.Nsensors):
            self.p_yi[s_idx] = np.sum(np.copy(self.p_yix[s_idx]),-1)

    def run_one_iteration(self):
        """template function to run one iteration of the ib algorithm"""
        pass

    def run(self):
        """template function to run the ib algorithm"""
        pass


