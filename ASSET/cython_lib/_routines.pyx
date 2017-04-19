# -*- coding: utf-8 -*-
#
# cython: cdivision=True
# cython: wraparound=False
#

import numpy as np
cimport numpy as np
import cython

cdef extern from "_routines.hpp":
    void expopt_array(float* input, int entries, float logK)


@cython.boundscheck(False)
def exp_opt_array(np.ndarray[float, ndim=2] input, logK):
    """
    return square of input

    :param input a double
    :return: square of input
    """
    input = np.ascontiguousarray(input)
    expopt_array(&input[0,0], 
                 input.shape[0] * input.shape[1],
                 logK)

