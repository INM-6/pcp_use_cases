# -*- coding: utf-8 -*-
#
# cython: cdivision=True
# cython: wraparound=False
#

import numpy as np
cimport numpy as np
import cython

# Floating point types to use with NumPy
#
# IEEE 754 says that float is 32 bits and double is 64 bits
#
ctypedef double dtype_float_t

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

