"""
Utility functions for qrevIPRD

Created by
Wylie Stroberg 2020-04-28
"""
import numpy as np

###########################################################################################
def wrapped_vector(dx,periodic_box=None):
    """ Correct position vectors accounting for periodic boundary conditions.
    """
    if periodic_box is not None:
        dx = np.where(dx > periodic_box * 0.5,dx-periodic_box,dx)
        dx = np.where(dx <= -periodic_box * 0.5,dx+periodic_box,dx)
    return dx
#-----------------------------------------------------------------
def dist(x1,x2,periodic_box=None):
    """ Calculate distance between two points using the minimum image
        convention to account for periodic boundary conditions.
    """
    dx = wrapped_vector(x2 - x1,periodic_box)
    return np.linalg.norm(dx,axis=1)

#-----------------------------------------------------------------
def gen_rand_vecs(number, dims=3):
    """ Generates number of random vectors of dimension dims.
    """
    rand_vecs = np.random.standard_normal(size=(number,dims))
    norms = np.linalg.norm(rand_vecs,axis=1)
    return rand_vecs/norms[:,None]

