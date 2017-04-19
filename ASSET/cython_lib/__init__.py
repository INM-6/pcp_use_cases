"""
Exposes routines implemented in compiled lib 
"""
from . import _routines


def fast_exp_array(input, logK):
    """
    Perform an element wise exponent calculation on all the elements in the 
    input. 
    The calculations is optimized for ELEPHANT ASSET data:
    If the value is -Inf or smaller in size then ~ 1E-10 the returned value is
    zero. If the data is non-contiguous the data is copied to one that is.
    Result value is written back into the numpy array.

    input:  array of float32 values (typically a numpy array)
    logK:   Controll value that sets the 'minimal' value of the probability.

    The functionality is extremely closely coupled to the ASSET implementation
    and is not expected to have reuse value.

    """
    _routines.exp_opt_array(input, logK)
