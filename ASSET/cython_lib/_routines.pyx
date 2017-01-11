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

    dtype_float_t squared_simple_cython(const dtype_float_t)


@cython.boundscheck(False)
def squared_simple(dtype_float_t input):
    """
    return square of input

    :param input a double
    :return: square of input
    """
    return squared_simple_cython(input)


cdef extern from "_routines.hpp":
    float logapprox(const float)
   


@cython.boundscheck(False)
def log_approx(float input):
    """
    return square of input

    :param input a double
    :return: square of input
    """
    return logapprox(input)

cdef extern from "_routines.hpp":
    float logapprox_array(float* input, int entries)


@cython.boundscheck(False)
def log_approx_array(np.ndarray[float, ndim=3] input):
    """
    return square of input

    :param input a double
    :return: square of input
    """
    input = np.ascontiguousarray(input)
    logapprox_array(&input[0,0,0], 
                    input.shape[0] * input.shape[1] * input.shape[2])

    return input




cdef extern from "_routines.hpp":
    float logapprox_multiply_array(float* input1, float* input2, int entries)


@cython.boundscheck(False)
def log_approx_multiply_array(np.ndarray[float, ndim=3] input1,
                     np.ndarray[float, ndim=3] input2):
    """
    return square of input

    :param input a double
    :return: square of input
    """
    input1 = np.ascontiguousarray(input1)
    input2 = np.ascontiguousarray(input2)
    logapprox_multiply_array(&input1[0,0,0], &input2[0,0,0], 
                    input1.shape[0] * input1.shape[1] * input1.shape[2])

    return input1



cdef extern from "_routines.hpp":
    float multiply_two_array(float* input1, float* input2, int entries)


@cython.boundscheck(False)
def multiply_arrays(np.ndarray[float, ndim=3] input1,
                    np.ndarray[float, ndim=3] input2):
    """
    return square of input

    :param input a double
    :return: square of input
    """
    input1 = np.ascontiguousarray(input1)
    input2 = np.ascontiguousarray(input2)
    multiply_two_array(&input1[0,0,0], 
                       &input2[0,0,0], 
                    input1.shape[0] * input1.shape[1] * input1.shape[2])

    return input1


cdef extern from "_routines.hpp":
    float expapprox_array(float* input, int entries)


@cython.boundscheck(False)
def exp_approx_array(np.ndarray[float, ndim=2] input):
    """
    return square of input

    :param input a double
    :return: square of input
    """
    input = np.ascontiguousarray(input)
    expapprox_array(&input[0,0], 
                    input.shape[0] * input.shape[1])

    return input

cdef extern from "_routines.hpp":
    float expapprox(float input)


@cython.boundscheck(False)
def exp_approx(float input):
    """
    return square of input

    :param input a double
    :return: square of input
    """
    expapprox(input)

    return expapprox(input)
