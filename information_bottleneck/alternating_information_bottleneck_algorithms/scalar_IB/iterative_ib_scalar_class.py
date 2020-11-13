import sys
import os
import numpy as np
import numpy.matlib

from generic_IB_scalar_class import GenericIBScalar

__author__ = "Steffen Steiner"
__copyright__ = "12.11.2018, Institute of Communications, University of Rostock "
__credits__ = ["Steffen Steiner"]
__version__ = "1.0"
__email__ = "steffen.steiner@uni-rostock.de"
__status__ = "Release"
__doc__="""This module contains the iterative Information Bottleneck algorithm for one sensor"""


class IterativeIBScalar(GenericIBScalar):
    """This class can be used to perform the iterative Information Bottleneck algorithm for a scalar sensor.
        Args:
    input parameter
        p_xy                    input joint pdf for x and y -> first dimension x, seccond dimension y
        p_init                  initial pmf p_z_y
        Nx                      cardinality of x
        Ny                      cardinality of y
        Nz                      cardinality of z
        beta
        accuracy                accuracy
        max_iter                maximum number of iterations of ib algorithm
    output PDF_s
        p_z_y
    """

    def __init__(self, p_xy_, p_init_, Nx_, Ny_, Nz_, beta_ , accuracy_, max_iter_):
        GenericIBScalar.__init__(self, p_xy_, p_init_, Nx_, Ny_, Nz_, beta_, accuracy_, max_iter_)
        self.name = 'Iterative IB for one sensor'

    def run_one_iteration(self,p_z_y,beta):

        p_y = np.sum(self.p_xy,0)
        p_yx = np.transpose(self.p_xy)
        p_x_y = p_yx / np.tile(np.expand_dims(p_y,-1),np.append(1,self.Nx))

        # uptdate quantizer output
        p_z = np.sum(p_z_y * np.tile(p_y,np.append(self.Nz, 1)),1)

        # update conditional pmf between x and z
        p_x_z = (1/np.tile(np.expand_dims(p_z,-1),np.append(1,self.Nx))) * np.sum(np.tile(np.expand_dims(p_yx,0),np.append(self.Nz,np.ones(2,dtype=int)))*np.tile(np.expand_dims(p_z_y,-1),np.append(np.ones(2,dtype=int),self.Nx)),1)

        # determine current Kullback-Leibler divergence
        KL = np.zeros((self.Nz,self.Ny),dtype=int)
        for runx in range(self.Nx):
            KL = KL + np.tile(p_x_y[:,runx],np.append(self.Nz,1)) * (np.tile(np.log(p_x_y[:,runx]),np.append(self.Nz,1)) - np.tile(np.expand_dims(np.log(p_x_z[:,runx]),-1),np.append(1,self.Ny)))

        # calculate conditional pmf of quantizer
        if np.isinf(beta):
            ptr = np.argmin(KL,0)
            p_z_y = np.zeros(np.append(self.Nz, self.Ny))
            p_z_y[ptr, np.arange(ptr.size,dtype=int)] = 1
        else:
            p_z_y = np.tile(np.expand_dims(p_z,-1),np.append(1,self.Ny)) * np.exp(-beta*KL)
            p_z_y = p_z_y / np.tile(np.expand_dims(np.sum(p_z_y,0),0),np.append(self.Nz,1))

        # update conditional pmf between x and z
        p_x_z = (1 / np.tile(np.expand_dims(p_z, -1), np.append(1, self.Nx))) * np.sum(np.tile(np.expand_dims(p_yx, 0), np.append(self.Nz, np.ones(2, dtype=int))) * np.tile(np.expand_dims(p_z_y, -1), np.append(np.ones(2, dtype=int), self.Nx)), 1)
        p_zx = p_x_z * np.tile(np.expand_dims(p_z,-1),np.append(1,self.Nx))
        p_z_x = p_zx / np.tile(np.expand_dims(self.p_x,0),np.append(self.Nz,1))

        return p_z_y, p_z_x

    # --------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------- helper functions ------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------

    #...
