import sys
import os
import numpy as np

__author__ = "Steffen Steiner"
__copyright__ = "14.11.2018, Institute of Communications, University of rostock"
__credits__ = ["Steffen Steiner"]
__version__ = "1.0"
__email__ = "steffen.steiner@uni-rostock.de"
__status__ = "Release"
__doc__ = """This module contains the definition of the abstract vectorized information bottleneck class multiple sensors.
          """


class GenericIBVector:
    """Common base class for all vector Information Bottleneck solver classes multiple sensors
        Args:
    input parameter
        p_xy                    input joint pdf
        p_init                  initial pmf p_z_y
        Nx                      cardinality of x
        Ny                      cardinality of y
        Nz                      cardinality of z
        beta
        accuracy                accuracy
        max_iter                maximum number of iterations of ib algorithm
    mutual information
        MI_XZ                   mutual information of output I(X;Z)
        MI_XY                   mutual information of input I(X;Y)
    output PDF_s
        p_z_y
        p_x_z
        p_z
    """

    def __init__(self,Nsensors_, p_xy_, p_init_, Nx_, Ny_, Nz_, beta_, accuracy_ , max_iter_):
        # initialize parameters
        self.Nsensors = Nsensors_
        self.p_xy = p_xy_
        self.p_y = np.sum(self.p_xy,0)
        self.p_x = np.sum(self.p_xy,tuple(1+np.arange(0,self.Nsensors,1,dtype=int)))
        self.p_init = p_init_
        self.Nx = Nx_
        self.Ny = Ny_
        self.Nz = Nz_
        self.beta = beta_
        self.accuracy = accuracy_
        self.max_iter = max_iter_

        # initialize unused parameters
        self.MI_XT = 0
        self.MI_XY = 0
        self.p_z_y = p_init_
        self.name = 'GenericIBVectorClass'


    def run_one_iteration(self):
        """template function to run one iteration of the ib algorithm"""
        pass

    def run(self):
        """template function to run the ib algorithm"""
        pass


